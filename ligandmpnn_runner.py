import json
import random
from pathlib import Path

import numpy as np
import torch
from ligandmpnn_code.data_utils import alphabet, featurize, parse_PDB
from ligandmpnn_code.model_utils import ProteinMPNN

# --- hard-coded optimization preset ---
MODEL_TYPE = "ligand_mpnn"
CHECKPOINT_LIGAND_MPNN = "./model_params/ligandmpnn_v_32_010_25.pt"
PDB_PATH_MULTI = "./optimization_set.json"
SEED = 111
BATCH_SIZE = 1
NUMBER_OF_BATCHES = 10
TEMPERATURE = 0.1
LIGAND_MPNN_CUTOFF_FOR_SCORE = 5.0
VERBOSE = 0

# kept fixed to the preset/default behavior used here
LIGAND_MPNN_USE_ATOM_CONTEXT = 1
LIGAND_MPNN_USE_SIDE_CHAIN_CONTEXT = 0
PARSE_ATOMS_WITH_ZERO_OCCUPANCY = 0


def run_optimization_and_compute_metrics():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(CHECKPOINT_LIGAND_MPNN, map_location=device)
    atom_context_num = checkpoint["atom_context_num"]
    k_neighbors = checkpoint["num_edges"]

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device=device,
        atom_context_num=atom_context_num,
        model_type=MODEL_TYPE,
        ligand_mpnn_use_side_chain_context=LIGAND_MPNN_USE_SIDE_CHAIN_CONTEXT,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with open(PDB_PATH_MULTI, "r") as fh:
        pdb_paths_raw = json.load(fh)
    pdb_paths = list(pdb_paths_raw)  # works if optimization_set.json is a dict or a list

    per_protein_mean_rec = []
    per_protein_details = []

    with torch.no_grad():
        for pdb in pdb_paths:
            name = Path(pdb).stem

            protein_dict, _, other_atoms, _, _ = parse_PDB(
                pdb,
                device=device,
                chains=[],
                parse_all_atoms=bool(LIGAND_MPNN_USE_SIDE_CHAIN_CONTEXT),
                parse_atoms_with_zero_occupancy=PARSE_ATOMS_WITH_ZERO_OCCUPANCY,
            )

            # Design all residues
            protein_dict["chain_mask"] = torch.ones(
                protein_dict["S"].shape[0],
                device=device,
                dtype=torch.int32,
            )

            if other_atoms:
                other_atoms.setBetas(other_atoms.getBetas() * 0.0)

            feature_dict = featurize(
                protein_dict,
                cutoff_for_score=LIGAND_MPNN_CUTOFF_FOR_SCORE,
                use_atom_context=LIGAND_MPNN_USE_ATOM_CONTEXT,
                number_of_ligand_atoms=atom_context_num,
                model_type=MODEL_TYPE,
            )
            feature_dict["batch_size"] = BATCH_SIZE
            _, L, _, _ = feature_dict["X"].shape
            feature_dict["temperature"] = TEMPERATURE
            feature_dict["bias"] = torch.zeros(
                (1, L, len(alphabet)),
                device=device,
                dtype=torch.float32,
            )
            feature_dict["symmetry_residues"] = [[]]
            feature_dict["symmetry_weights"] = [[]]

            S_list = []
            for _ in range(NUMBER_OF_BATCHES):
                feature_dict["randn"] = torch.randn(
                    (BATCH_SIZE, feature_dict["mask"].shape[1]),
                    device=device,
                )
                output_dict = model.sample(feature_dict)
                S_list.append(output_dict["S"])

            S_stack = torch.cat(S_list, dim=0)                   # [N, L]
            S_native = feature_dict["S"][:1].expand_as(S_stack) # [N, L]

            ligand_mask = (
                feature_dict["mask"][:1]
                * feature_dict["mask_XY"][:1]
                * feature_dict["chain_mask"][:1]
            ).expand_as(S_stack)

            n_ligand_res = int(ligand_mask[0].sum().item())
            if n_ligand_res == 0:
                raise RuntimeError(f"{name}: no ligand-proximal residues found")

            seq_rec = ((S_native == S_stack).float() * ligand_mask).sum(-1) / (
                ligand_mask.sum(-1) + 1e-8
            )

            mean_rec = float(seq_rec.mean().item())
            per_protein_mean_rec.append(mean_rec)
            per_protein_details.append((name, mean_rec, n_ligand_res))

            print(
                f"{name}: mean ligand-proximal seq recovery = {100.0 * mean_rec:.1f}% "
                f"(n_ligand_res={n_ligand_res}, n_seqs={S_stack.shape[0]})"
            )

    print(f"\nEvaluated: {len(per_protein_mean_rec)} proteins")

    if not per_protein_mean_rec:
        raise RuntimeError("No proteins evaluated.")

    arr = np.array(per_protein_mean_rec, dtype=float)
    median_rec = float(np.median(arr))
    mean_rec = float(np.mean(arr))

    print("\n" + "=" * 60)
    print(
        f"Median sequence recovery (ligand-proximal, {LIGAND_MPNN_CUTOFF_FOR_SCORE:.1f}A): "
        f"{100.0 * median_rec:.1f}%"
    )
    print(
        f"Mean   sequence recovery (ligand-proximal, {LIGAND_MPNN_CUTOFF_FOR_SCORE:.1f}A): "
        f"{100.0 * mean_rec:.1f}%"
    )
    print("=" * 60)

    return {
        "per_protein_details": per_protein_details,
        "median_recovery": median_rec,
        "mean_recovery": mean_rec,
    }


results = run_optimization_and_compute_metrics()
val_metric = results["mean_recovery"]
print(f"tuso_evaluate:{val_metric}")