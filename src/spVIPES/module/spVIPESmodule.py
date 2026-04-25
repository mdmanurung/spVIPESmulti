"""Main module."""
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
import zuko.flows
from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomialMixture
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi.nn import FCLayers
from spVIPES.nn.networks import Encoder, LinearDecoderSPVIPE
from spVIPES.module.utils import gradient_reversal

torch.backends.cudnn.benchmark = True


class spVIPESmodule(BaseModuleClass):
    """
    PyTorch implementation of spVIPES variational autoencoder module.

    This module implements the core variational autoencoder with Product of Experts (PoE)
    for shared-private latent space learning. It extends scVI's underlying VAE architecture
    with multi-group integration capabilities and support for different PoE strategies.

    Parameters
    ----------
    groups_lengths : list of int
        List containing the number of features (genes) for each group/dataset.
    groups_obs_names : list
        List of observation names for each group.
    groups_var_names : list
        List of variable (gene) names for each group.
    groups_obs_indices : list
        List of observation indices for each group.
    groups_var_indices : list
        List of variable indices for each group.
    transport_plan : torch.Tensor, optional
        Precomputed optimal transport plan matrix for PoE alignment.
    pair_data : bool, default=False
        Whether to use paired data for direct cell-to-cell correspondences.
    use_labels : bool, default=False
        Whether to use cell type labels for supervised PoE alignment.
    n_labels : int, optional
        Number of unique cell type labels when using supervised alignment.
    n_batch : int, default=0
        Number of batches. If 0, no batch correction is performed.
    n_hidden : int, default=128
        Number of nodes per hidden layer in encoder and decoder networks.
    n_dimensions_shared : int, default=25
        Dimensionality of the shared latent space capturing common features.
    n_dimensions_private : int, default=10
        Dimensionality of private latent spaces capturing group-specific features.
    dropout_rate : float, default=0.1
        Dropout rate for neural networks to prevent overfitting.
    use_batch_norm : bool, default=True
        Whether to use batch normalization in neural networks.
    use_layer_norm : bool, default=False
        Whether to use layer normalization in neural networks.
    log_variational_inference : bool, default=True
        Whether to log-transform data before encoding for numerical stability.
    log_variational_generative : bool, default=True
        Whether to log-transform data before decoding for numerical stability.
    dispersion : {"gene", "gene-batch", "gene-cell"}, default="gene"
        Level at which to model the dispersion parameter in the negative binomial distribution.

    Notes
    -----
    This module is based on the scVI framework and implements the variational inference
    described in the spVIPES paper. The Product of Experts mechanism allows for flexible
    integration of multiple single-cell datasets with different feature sets.
    """

    def __init__(
        self,
        groups_lengths,
        groups_obs_names,
        groups_var_names,
        groups_obs_indices,
        groups_var_indices,
        transport_plan: Optional[torch.Tensor] = None,
        pair_data: bool = False,
        use_labels: bool = False,
        n_labels: Optional[int] = None,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_dimensions_shared: int = 25,
        n_dimensions_private: int = 10,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        log_variational_inference: bool = True,
        log_variational_generative: bool = True,
        dispersion: Literal["gene", "gene-batch", "gene-cell"] = "gene",
        # Multimodal parameters
        groups_modality_lengths: Optional[dict] = None,
        groups_modality_var_indices: Optional[dict] = None,
        modality_likelihoods: Optional[dict[str, str]] = None,
        modality_names: Optional[list[str]] = None,
        # Normalizing flow prior parameters
        use_nf_prior: bool = False,
        nf_type: Literal["NSF", "MAF"] = "NSF",
        nf_transforms: int = 3,
        nf_bins: int = 8,
        nf_target: Literal["shared", "private", "both"] = "shared",
        # Disentanglement objective parameters (CellDISECT / Multi-ContrastiveVAE)
        disentangle_group_shared_weight: float = 0.0,
        disentangle_label_shared_weight: float = 0.0,
        disentangle_group_private_weight: float = 0.0,
        disentangle_label_private_weight: float = 0.0,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.1,
    ):
        """
        Initialize the spVIPES variational autoencoder module.

        This method sets up the neural network components including encoders and decoders
        for each group, and configures the Product of Experts mechanism based on the
        provided parameters. The module extends scVI's VAE architecture for multi-group
        integration with shared-private latent spaces.

        Notes
        -----
        The initialization automatically configures the appropriate PoE strategy based on
        the provided inputs (transport_plan, use_labels, pair_data). The module creates
        separate encoders and decoders for each group while sharing the latent space
        structure for integration.

        For multimodal data, provide ``groups_modality_lengths``, ``groups_modality_var_indices``,
        ``modality_likelihoods``, and ``modality_names``. Each (group, modality) pair gets
        its own encoder and decoder, with a two-level PoE: intra-group across modalities,
        then inter-group.
        """
        super().__init__()
        self.n_dimensions_shared = n_dimensions_shared
        self.n_dimensions_private = n_dimensions_private
        self.n_batch = n_batch
        self.input_dims = groups_lengths
        self.groups_barcodes = groups_obs_names
        self.groups_genes = groups_var_names
        self.groups_obs_indices = groups_obs_indices
        self.groups_var_indices = groups_var_indices
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.label_per_batch = []
        self.dispersion = dispersion
        self.log_variational_inference = log_variational_inference
        self.log_variational_generative = log_variational_generative

        # Multimodal configuration
        self.is_multimodal = groups_modality_lengths is not None
        self.groups_modality_lengths = groups_modality_lengths
        self.groups_modality_var_indices = groups_modality_var_indices
        self.modality_likelihoods = modality_likelihoods or {}
        self.modality_names = modality_names or []
        # Track which modalities each group has
        if self.is_multimodal:
            self.group_modalities = {g: list(mod_dict.keys()) for g, mod_dict in groups_modality_lengths.items()}
        else:
            self.group_modalities = None

        cat_list = [n_batch] if n_batch > 0 else None

        if self.is_multimodal:
            # Multimodal mode: per-(group, modality) encoders and decoders
            self.px_r = torch.nn.ParameterDict()
            self.encoders = {}
            self.decoders = {}

            for group, mod_dict in groups_modality_lengths.items():
                for modality, n_features in mod_dict.items():
                    key = f"{group}_{modality}"
                    self.px_r[key] = torch.nn.Parameter(torch.randn(n_features))

                    self.encoders[(group, modality)] = {
                        "shared": Encoder(
                            n_features, n_dimensions_shared,
                            hidden=n_hidden, dropout=dropout_rate,
                            n_cat_list=cat_list, groups=group,
                        ),
                        "private": Encoder(
                            n_features, n_dimensions_private,
                            hidden=n_hidden, dropout=dropout_rate,
                            n_cat_list=cat_list, groups=group,
                        ),
                    }
                    self.decoders[(group, modality)] = LinearDecoderSPVIPE(
                        n_dimensions_private, n_dimensions_shared, n_features,
                        n_cat_list=cat_list, use_batch_norm=True,
                        use_layer_norm=False, bias=False,
                    )

            # Register sub-modules
            for (group, modality), enc_dict in self.encoders.items():
                self.add_module(f"encoder_{group}_{modality}_shared", enc_dict["shared"])
                self.add_module(f"encoder_{group}_{modality}_private", enc_dict["private"])
            for (group, modality), dec in self.decoders.items():
                self.add_module(f"decoder_{group}_{modality}", dec)
        else:
            # Single-modality mode (backward compatible)
            self.px_r = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.randn(length)) for length in groups_lengths.values()]
            )
            self.encoders = {
                groups: {
                    "shared": Encoder(
                        x_dim, n_dimensions_shared,
                        hidden=n_hidden, dropout=dropout_rate,
                        n_cat_list=cat_list, groups=groups,
                    ),
                    "private": Encoder(
                        x_dim, n_dimensions_private,
                        hidden=n_hidden, dropout=dropout_rate,
                        n_cat_list=cat_list, groups=groups,
                    ),
                }
                for groups, x_dim in self.input_dims.items()
            }
            self.decoders = {
                groups: LinearDecoderSPVIPE(
                    n_dimensions_private, n_dimensions_shared, x_dim,
                    n_cat_list=cat_list, use_batch_norm=True,
                    use_layer_norm=False, bias=False,
                )
                for groups, x_dim in self.input_dims.items()
            }

            # Register sub-modules
            for (groups, values_encoder), (_, values_decoder) in zip(self.encoders.items(), self.decoders.items()):
                self.add_module(f"encoder_{groups}_shared", values_encoder["shared"])
                self.add_module(f"encoder_{groups}_private", values_encoder["private"])
                self.add_module(f"decoder_{groups}", values_decoder)

        # Store the transport plan as an attribute
        self.use_transport_plan = transport_plan is not None
        self.transport_plan = transport_plan
        self.use_labels = use_labels
        self.n_labels = n_labels
        self.pair_data = pair_data

        # Normalizing flow prior
        self.use_nf_prior = use_nf_prior
        self.nf_target = nf_target
        if use_nf_prior:
            flow_cls = zuko.flows.NSF if nf_type == "NSF" else zuko.flows.MAF
            flow_kwargs = {"transforms": nf_transforms}
            if nf_type == "NSF":
                flow_kwargs["bins"] = nf_bins

            if nf_target in ("shared", "both"):
                self.flow_prior_shared = flow_cls(features=n_dimensions_shared, context=0, **flow_kwargs)
            if nf_target in ("private", "both"):
                self.flow_prior_private = flow_cls(features=n_dimensions_private, context=0, **flow_kwargs)

        # Disentanglement objective: 4 auxiliary classifiers (CellDISECT-style).
        # Note: this is NOT the MIG metric (Chen et al. 2018) — these are a mix
        # of adversarial domain-invariance losses (GRL) and supervised CE losses
        # acting as variational MI lower bounds. Group classifiers (q_group_*)
        # require only group identity (always known); label classifiers and
        # contrastive require use_labels=True.
        _label_required = (
            ("disentangle_label_shared_weight", disentangle_label_shared_weight),
            ("disentangle_label_private_weight", disentangle_label_private_weight),
            ("contrastive_weight", contrastive_weight),
        )
        _violations = [name for name, w in _label_required if w > 0]
        if _violations and not use_labels:
            raise ValueError(
                f"The following disentanglement/contrastive weights require use_labels=True "
                f"(label-based PoE): {_violations}. Either provide a label_key "
                f"in setup_anndata() or set those weights to 0.0. Group classifiers "
                f"(q_group_shared, q_group_private) do not require labels and "
                f"remain enabled."
            )

        n_groups = len(groups_lengths)
        self.disentangle_group_shared_weight = disentangle_group_shared_weight
        self.disentangle_label_shared_weight = disentangle_label_shared_weight
        self.disentangle_group_private_weight = disentangle_group_private_weight
        self.disentangle_label_private_weight = disentangle_label_private_weight
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature

        _clf_kwargs = dict(n_layers=2, n_hidden=64, dropout_rate=0.1, use_batch_norm=True)

        # Classifier 2: adversarial — erase group info from z_shared
        self.q_group_shared = (
            FCLayers(n_in=n_dimensions_shared, n_out=n_groups, **_clf_kwargs)
            if disentangle_group_shared_weight > 0 else None
        )
        # Classifier 1: supervised — preserve label info in z_shared
        self.q_label_shared = (
            FCLayers(n_in=n_dimensions_shared, n_out=n_labels, **_clf_kwargs)
            if disentangle_label_shared_weight > 0 and use_labels else None
        )
        # Classifier 3: supervised — preserve group info in z_private
        self.q_group_private = (
            FCLayers(n_in=n_dimensions_private, n_out=n_groups, **_clf_kwargs)
            if disentangle_group_private_weight > 0 else None
        )
        # Classifier 4: adversarial — erase label info from z_private
        self.q_label_private = (
            FCLayers(n_in=n_dimensions_private, n_out=n_labels, **_clf_kwargs)
            if disentangle_label_private_weight > 0 and use_labels else None
        )

        # Optional prototype buffer for contrastive InfoNCE on z_shared
        self.prototypes = None
        if contrastive_weight > 0 and use_labels:
            self.register_buffer("prototypes", torch.zeros(n_groups, n_labels, n_dimensions_shared))
            self.prototype_momentum = 0.99


    def _cluster_based_poe(
        self, shared_stats: dict, batch_transport_plans: dict[int, torch.Tensor], processed_labels: list[torch.Tensor]
    ):
        if len(shared_stats) != 2:
            raise ValueError(
                f"Cluster-based PoE only supports exactly 2 groups, got {len(shared_stats)}. "
                "Use label-based PoE for more than 2 groups."
            )
        groups_1_stats, groups_2_stats = shared_stats.values()
        groups_1_stats = {
            k: groups_1_stats[k] for k in ["logtheta_loc", "logtheta_logvar", "logtheta_scale"] if k in groups_1_stats
        }
        groups_2_stats = {
            k: groups_2_stats[k] for k in ["logtheta_loc", "logtheta_logvar", "logtheta_scale"] if k in groups_2_stats
        }

        # The processed_labels are already batched, so we can use them directly
        batch_labels_1 = processed_labels[0]  # Labels for group 1
        batch_labels_2 = processed_labels[1]  # Labels for group 2

        poe_stats_per_component = {}
        unique_components = torch.unique(torch.cat([batch_labels_1, batch_labels_2]))
        for component in unique_components:
            mask_1 = (batch_labels_1 == component).squeeze()
            mask_2 = (batch_labels_2 == component).squeeze()

            if torch.any(mask_1) and torch.any(mask_2):
                # Extract the relevant part of the batch transport plan for each dataset
                component_plan_1 = batch_transport_plans[0][mask_1][:, mask_2]
                component_plan_2 = batch_transport_plans[1][mask_2][:, mask_1]

                # Normalize the component plans while preserving zeros
                def normalize_plan(plan):
                    row_sums = plan.sum(dim=1, keepdim=True)
                    row_sums = row_sums.clamp(min=1e-10)  # Avoid division by zero
                    return torch.where(plan > 0, plan / row_sums, plan)

                normalized_plan_1 = normalize_plan(component_plan_1)
                normalized_plan_2 = normalize_plan(component_plan_2)

                # Compute weighted average for group 1
                component_stats_1 = {}
                for k, v in groups_1_stats.items():
                    weighted_v = torch.matmul(normalized_plan_1, v[mask_2])
                    component_stats_1[k] = weighted_v

                # Compute weighted average for group 2
                component_stats_2 = {}
                for k, v in groups_2_stats.items():
                    weighted_v = torch.matmul(normalized_plan_2, v[mask_1])
                    component_stats_2[k] = weighted_v

                # Perform PoE
                poe_stats_per_component[component.item()] = self._poe_n({0: component_stats_1, 1: component_stats_2})
            else:
                # Handle unmatched cells
                if torch.any(mask_1):
                    poe_stats_per_component[component.item()] = {
                        0: {k: v[mask_1] for k, v in groups_1_stats.items()},
                        1: {k: torch.empty((0, v.shape[1]), device=v.device) for k, v in groups_2_stats.items()},
                    }
                if torch.any(mask_2):
                    poe_stats_per_component[component.item()] = {
                        0: {k: torch.empty((0, v.shape[1]), device=v.device) for k, v in groups_1_stats.items()},
                        1: {k: v[mask_2] for k, v in groups_2_stats.items()},
                    }

        # Initialize the output tensors
        groups_1_output = {
            k: torch.empty(groups_1_stats[k].shape, dtype=torch.float32, device=groups_1_stats[k].device)
            for k in groups_1_stats
        }
        groups_2_output = {
            k: torch.empty(groups_2_stats[k].shape, dtype=torch.float32, device=groups_2_stats[k].device)
            for k in groups_2_stats
        }

        # Fill the output tensors while maintaining the original cell order
        for group, labels, output in [(0, batch_labels_1, groups_1_output), (1, batch_labels_2, groups_2_output)]:
            component_count = {}
            for i, component in enumerate(labels):
                component = component.item()
                count = component_count.get(component, 0)
                component_count[component] = count + 1

                component_stats = poe_stats_per_component[component][group]
                tensor_index = count % component_stats["logtheta_loc"].size(0)

                for k in output:
                    output[k][i] = component_stats[k][tensor_index]

        concat_poe_stats = {0: groups_1_output, 1: groups_2_output}

        # Compute qz and theta for both groups
        for group in [0, 1]:
            concat_poe_stats[group]["logtheta_qz"] = Normal(
                concat_poe_stats[group]["logtheta_loc"], concat_poe_stats[group]["logtheta_scale"].clamp(min=1e-6)
            )
            concat_poe_stats[group]["logtheta_log_z"] = concat_poe_stats[group]["logtheta_qz"].rsample()
            concat_poe_stats[group]["logtheta_theta"] = F.softmax(concat_poe_stats[group]["logtheta_log_z"], -1)

        return concat_poe_stats

    def _poe_n(self, shared_stats: dict):
        """Generic N-group Product of Experts.

        Combines shared statistics from N >= 2 groups using Gaussian PoE.
        Handles unequal batch sizes by padding with standard normal prior
        (loc=0, logvar=0 → var=1, precision=1).

        Parameters
        ----------
        shared_stats : dict[int, dict[str, Tensor]]
            Per-group encoder statistics with keys "logtheta_loc", "logtheta_logvar", "logtheta_scale".

        Returns
        -------
        dict[int, dict[str, Tensor]]
            Per-group PoE results with keys "logtheta_loc", "logtheta_logvar", "logtheta_scale".
        """
        group_keys = sorted(shared_stats.keys())
        n_groups = len(group_keys)
        if n_groups < 2:
            raise ValueError(f"PoE requires at least 2 groups, got {n_groups}")

        group_sizes = {g: shared_stats[g]["logtheta_logvar"].shape[0] for g in group_keys}
        max_batch_size = max(group_sizes.values())
        latent_dim = shared_stats[group_keys[0]]["logtheta_logvar"].shape[1]
        device = shared_stats[group_keys[0]]["logtheta_logvar"].device

        # Pad each group to max_batch_size and stack
        # Padding: loc=0, logvar=0 → var=1, 1/var=1 (standard normal prior contribution)
        padded_locs = []
        padded_logvars = []
        for g in group_keys:
            loc = shared_stats[g]["logtheta_loc"]
            logvar = shared_stats[g]["logtheta_logvar"]
            g_size = group_sizes[g]
            if g_size < max_batch_size:
                pad_size = max_batch_size - g_size
                loc = torch.cat([loc, torch.zeros(pad_size, latent_dim, device=device)], dim=0)
                logvar = torch.cat([logvar, torch.zeros(pad_size, latent_dim, device=device)], dim=0)
            padded_locs.append(loc)
            padded_logvars.append(logvar)

        # Stack: shape (N, max_batch, latent_dim)
        stacked_mus = torch.stack(padded_locs, dim=0)
        stacked_logvars = torch.stack(padded_logvars, dim=0)

        # Compute joint PoE
        mus_joint, logvars_joint = self._product_of_experts(stacked_mus, stacked_logvars)

        # Slice back to each group's original size and build result
        result = {}
        for g in group_keys:
            g_size = group_sizes[g]
            g_mu = mus_joint[:g_size]
            g_logvar = logvars_joint[:g_size]
            g_scale = torch.sqrt(torch.exp(g_logvar))
            result[g] = {
                "logtheta_loc": g_mu,
                "logtheta_logvar": g_logvar,
                "logtheta_scale": g_scale,
            }

        return result

    def _get_inference_input(self, tensors_by_group):
        x = {i: group[REGISTRY_KEYS.X_KEY] for i, group in enumerate(tensors_by_group)}
        batch_index = [group[REGISTRY_KEYS.BATCH_KEY] for group in tensors_by_group]
        groups = [group["groups"] for group in tensors_by_group]
        global_indices = [group["indices"] for group in tensors_by_group]

        input_dict = {
            "x": x,
            "batch_index": batch_index,
            "groups": groups,
            "global_indices": global_indices,
        }

        if self.use_transport_plan and not self.pair_data:
            required_key = "processed_transport_labels"
            if required_key not in tensors_by_group[0]:
                raise ValueError(f"{required_key} are required when using transport plan.")
            input_dict["processed_labels"] = [group[required_key] for group in tensors_by_group]

        if self.use_labels:
            if "labels" not in tensors_by_group[0]:
                raise ValueError("Labels are required when using label-based POE.")
            input_dict["labels"] = [group["labels"].flatten() for group in tensors_by_group]

        return input_dict

    def _get_generative_input(self, tensors_by_group, inference_outputs):
        private_stats = inference_outputs["private_stats"]
        shared_stats = inference_outputs["shared_stats"]
        poe_stats = inference_outputs["poe_stats"]
        library = inference_outputs["library"]
        groups = [group["groups"] for group in tensors_by_group]
        batch_index = [group[REGISTRY_KEYS.BATCH_KEY] for group in tensors_by_group]

        input_dict = {
            "private_stats": private_stats,
            "shared_stats": shared_stats,
            "poe_stats": poe_stats,
            "library": library,
            "groups": groups,
            "batch_index": batch_index,
        }

        # Pass through multimodal-specific data
        if "per_modality_private" in inference_outputs:
            input_dict["per_modality_private"] = inference_outputs["per_modality_private"]

        return input_dict

    @auto_move_data
    def inference(self, x, batch_index, groups, global_indices, **kwargs):
        """Runs the encoder model.

        In single-modality mode, runs per-group shared/private encoders then inter-group PoE.
        In multimodal mode, runs per-(group, modality) encoders, then intra-group PoE across
        modalities, then inter-group PoE.
        """
        if self.is_multimodal:
            return self._inference_multimodal(x, batch_index, groups, global_indices, **kwargs)

        # Single-modality mode (backward compatible)
        x = {
            i: xs[:, self.groups_var_indices[i]] for i, xs in x.items()
        }  # update each groups minibatch with its own gene indices

        if self.log_variational_inference:
            x = {i: torch.log(1 + xs) for i, xs in x.items()}  # logvariational

        library = {i: torch.log(xs.sum(1)).unsqueeze(1) for i, xs in x.items()}  # observed library size

        private_stats = {}
        shared_stats = {}

        for group, (item, batch) in enumerate(zip(x.values(), batch_index)):
            private_encoder = self.encoders[group]["private"]
            shared_encoder = self.encoders[group]["shared"]

            private_values = private_encoder(item, group, batch)
            shared_values = shared_encoder(item, group, batch)

            private_stats[group] = private_values
            shared_stats[group] = shared_values

        batch_transport_plans = None
        processed_labels = None
        labels = None

        if self.use_transport_plan:
            batch_transport_plans = self._get_batch_transport_plans(global_indices)
            if self.transport_plan is not None and not self.pair_data:
                processed_labels = kwargs.get("processed_labels")

        if self.use_labels:
            if "labels" in kwargs:
                labels = dict(enumerate(kwargs["labels"]))

        poe_stats = self._supervised_poe(shared_stats, batch_transport_plans, processed_labels, labels)

        outputs = {
            "private_stats": private_stats,
            "shared_stats": shared_stats,
            "poe_stats": poe_stats,
            "library": library,
        }

        return outputs

    def _inference_multimodal(self, x, batch_index, groups, global_indices, **kwargs):
        """Multimodal inference with two-level PoE."""
        n_groups = len(x)

        # Step 1: Per-(group, modality) encoding
        per_modality_private = {}  # keyed by (group, modality)
        per_modality_shared = {}   # keyed by (group, modality)
        library = {}               # keyed by (group, modality)

        for group in range(n_groups):
            x_group = x[group]  # full concatenated features for this group
            batch = batch_index[group]

            for modality in self.group_modalities[group]:
                # Slice the group's data to this modality's features
                mod_var_indices = self.groups_modality_var_indices[group][modality]
                x_mod = x_group[:, mod_var_indices]

                # Modality-specific preprocessing
                likelihood = self.modality_likelihoods.get(modality, "nb")
                if likelihood == "nb" and self.log_variational_inference:
                    x_mod_enc = torch.log(1 + x_mod)
                else:
                    x_mod_enc = x_mod  # Gaussian: data already log-normalized

                lib = torch.log(x_mod.sum(1).clamp(min=1)).unsqueeze(1)
                library[(group, modality)] = lib

                shared_enc = self.encoders[(group, modality)]["shared"]
                private_enc = self.encoders[(group, modality)]["private"]

                per_modality_shared[(group, modality)] = shared_enc(x_mod_enc, group, batch)
                per_modality_private[(group, modality)] = private_enc(x_mod_enc, group, batch)

        # Step 2: Intra-group PoE across modalities → per-group shared stats
        per_group_shared = {}
        for group in range(n_groups):
            modalities = self.group_modalities[group]
            if len(modalities) == 1:
                # Single modality in this group: use its shared stats directly
                mod = modalities[0]
                per_group_shared[group] = per_modality_shared[(group, mod)]
            else:
                # Multiple modalities: combine via PoE
                mod_shared = {}
                for idx, mod in enumerate(modalities):
                    stats = per_modality_shared[(group, mod)]
                    mod_shared[idx] = {
                        "logtheta_loc": stats["logtheta_loc"],
                        "logtheta_logvar": stats["logtheta_logvar"],
                        "logtheta_scale": stats["logtheta_scale"],
                    }
                intra_poe = self._poe_n(mod_shared)
                # Use index 0 result (all modalities have same batch size within a group)
                intra_mu = intra_poe[0]["logtheta_loc"]
                intra_logvar = intra_poe[0]["logtheta_logvar"]
                intra_scale = intra_poe[0]["logtheta_scale"]

                # Build full stats dict compatible with downstream
                qz = Normal(intra_mu, intra_scale.clamp(min=1e-6))
                log_z = qz.rsample()
                theta = F.softmax(log_z, -1)
                per_group_shared[group] = {
                    "logtheta_loc": intra_mu,
                    "logtheta_logvar": intra_logvar,
                    "logtheta_scale": intra_scale,
                    "log_z": log_z,
                    "theta": theta,
                    "qz": qz,
                }

        # Step 3: Inter-group PoE (same as single-modality)
        labels = None
        if self.use_labels and "labels" in kwargs:
            labels = dict(enumerate(kwargs["labels"]))

        poe_stats = self._supervised_poe(per_group_shared, None, None, labels)

        # Build output: private_stats keyed by group (use first modality for backward compat)
        # For multimodal, we also include per-modality private stats
        private_stats = {}
        for group in range(n_groups):
            # Use first modality's private stats as the group-level private
            first_mod = self.group_modalities[group][0]
            private_stats[group] = per_modality_private[(group, first_mod)]

        outputs = {
            "private_stats": private_stats,
            "shared_stats": per_group_shared,
            "poe_stats": poe_stats,
            "library": library,
            "per_modality_private": per_modality_private,
            "per_modality_shared": per_modality_shared,
        }

        return outputs

    def _get_batch_transport_plans(self, global_indices):
        if len(global_indices) != 2:
            raise ValueError(
                f"Transport plan-based PoE only supports exactly 2 groups, got {len(global_indices)}. "
                "Use label-based PoE for more than 2 groups."
            )
        # Convert to CPU numpy arrays if they're on GPU
        indices1 = global_indices[0].cpu().numpy() if isinstance(global_indices[0], torch.Tensor) else global_indices[0]
        indices2 = global_indices[1].cpu().numpy() if isinstance(global_indices[1], torch.Tensor) else global_indices[1]

        # Slice the transport plan for the current minibatches
        batch_transport_plan = self.transport_plan[indices1.squeeze()][:, indices2.squeeze()]

        return {0: batch_transport_plan, 1: batch_transport_plan.T}

    def _supervised_poe(
        self,
        shared_stats: dict,
        batch_transport_plans: Optional[dict[int, torch.Tensor]],
        processed_labels: Optional[list[torch.Tensor]],
        labels: Optional[dict[int, torch.Tensor]],
    ):
        # Prioritize label-based PoE when labels are explicitly provided
        if self.use_labels and labels is not None:
            return self._label_based_poe(shared_stats, labels)
        elif self.use_transport_plan:
            if self.pair_data:
                # Assuming batch_transport_plans[0] contains the transport plan for paired data
                return self._paired_poe(shared_stats, batch_transport_plans[0])
            elif batch_transport_plans is not None:
                if processed_labels is None:
                    raise ValueError("Processed labels are required when using transport plan.")
                # Convert processed_labels list to a dictionary
                label_group = {i: processed_labels[i] for i in range(len(processed_labels))}
                return self._cluster_based_poe(shared_stats, batch_transport_plans, label_group)
            else:
                raise ValueError(
                    "Either paired cells or batch transport plans must be provided when using transport plan."
                )
        else:
            raise ValueError("Either transport plan or labels must be provided for supervised POE.")

    def _paired_poe(self, shared_stats: dict, transport_plan: torch.Tensor):
        if len(shared_stats) != 2:
            raise ValueError(
                f"Paired PoE only supports exactly 2 groups, got {len(shared_stats)}. "
                "Use label-based PoE for more than 2 groups."
            )
        groups_1_stats, groups_2_stats = shared_stats.values()
        groups_1_stats = {
            k: groups_1_stats[k] for k in ["logtheta_loc", "logtheta_logvar", "logtheta_scale"] if k in groups_1_stats
        }
        groups_2_stats = {
            k: groups_2_stats[k] for k in ["logtheta_loc", "logtheta_logvar", "logtheta_scale"] if k in groups_2_stats
        }

        # Ensure both groups have the same number of cells
        assert (
            groups_1_stats["logtheta_loc"].shape[0] == groups_2_stats["logtheta_loc"].shape[0]
        ), "Paired PoE requires equal number of cells from both groups"

        # Find the index of the maximum value for each row in the transport plan
        max_indices_1to2 = torch.argmax(transport_plan, dim=1)
        max_indices_2to1 = torch.argmax(transport_plan, dim=0)

        # Use these indices to select the corresponding cells from the other dataset
        matched_stats_1 = {}
        matched_stats_2 = {}
        for k in groups_1_stats:
            matched_stats_1[k] = groups_2_stats[k][max_indices_1to2]
            matched_stats_2[k] = groups_1_stats[k][max_indices_2to1]

        # Compute joint statistics for group 1
        mus_1 = torch.stack([groups_1_stats["logtheta_loc"], matched_stats_1["logtheta_loc"]], dim=0)
        logvars_1 = torch.stack([groups_1_stats["logtheta_logvar"], matched_stats_1["logtheta_logvar"]], dim=0)
        mus_joint_1, logvars_joint_1 = self._product_of_experts(mus_1, logvars_1)

        # Compute joint statistics for group 2
        mus_2 = torch.stack([matched_stats_2["logtheta_loc"], groups_2_stats["logtheta_loc"]], dim=0)
        logvars_2 = torch.stack([matched_stats_2["logtheta_logvar"], groups_2_stats["logtheta_logvar"]], dim=0)
        mus_joint_2, logvars_joint_2 = self._product_of_experts(mus_2, logvars_2)

        # Compute scales from logvars
        scale_joint_1 = torch.exp(0.5 * logvars_joint_1)
        scale_joint_2 = torch.exp(0.5 * logvars_joint_2)

        poe_stats = {
            0: {
                "logtheta_loc": mus_joint_1,
                "logtheta_logvar": logvars_joint_1,
                "logtheta_scale": scale_joint_1,
            },
            1: {
                "logtheta_loc": mus_joint_2,
                "logtheta_logvar": logvars_joint_2,
                "logtheta_scale": scale_joint_2,
            },
        }

        # Compute qz and theta for both groups
        for group in [0, 1]:
            poe_stats[group]["logtheta_qz"] = Normal(
                poe_stats[group]["logtheta_loc"], poe_stats[group]["logtheta_scale"].clamp(min=1e-6)
            )
            poe_stats[group]["logtheta_log_z"] = poe_stats[group]["logtheta_qz"].rsample()
            poe_stats[group]["logtheta_theta"] = F.softmax(poe_stats[group]["logtheta_log_z"], -1)

        return poe_stats

    def _product_of_experts(self, mus, logvars):
        vars = torch.exp(logvars)
        mus_joint = torch.sum(mus / vars, dim=0)
        logvars_joint = torch.ones_like(mus_joint)
        logvars_joint += torch.sum(1.0 / vars, dim=0)
        logvars_joint = 1.0 / logvars_joint  # inverse
        mus_joint *= logvars_joint
        logvars_joint = torch.log(logvars_joint)
        return mus_joint, logvars_joint

    def _nf_kl(self, qz, z, target: str):
        """Compute KL divergence using normalizing flow prior via Monte Carlo.

        KL(q(z|x) || p_flow(z)) ≈ log q(z|x) - log p_flow(z)

        Parameters
        ----------
        qz : Normal
            Posterior distribution q(z|x).
        z : torch.Tensor
            Samples from q(z|x).
        target : str
            Which flow to use: "shared" or "private".

        Returns
        -------
        torch.Tensor
            KL divergence per sample, shape (batch_size,).
        """
        if target == "shared":
            flow_dist = self.flow_prior_shared()
        else:
            flow_dist = self.flow_prior_private()
        log_qz = qz.log_prob(z).sum(dim=-1)  # sum over latent dims
        log_pz = flow_dist.log_prob(z)  # flow gives scalar per sample
        return log_qz - log_pz

    def _label_based_poe(self, shared_stats: dict, label_group: dict):
        """Label-based PoE for N >= 2 groups.

        For each cell type label, combines shared encoder statistics from all groups
        that contain cells of that type using Product of Experts. Groups missing a label
        contribute an uninformative prior (loc=0, logvar=log(1)=0).
        """
        stat_keys = ["logtheta_loc", "logtheta_logvar", "logtheta_scale"]
        group_keys = sorted(shared_stats.keys())
        n_groups = len(group_keys)

        # Extract per-group stats and labels
        per_group_stats = {}
        per_group_labels = {}
        for g in group_keys:
            per_group_stats[g] = {k: shared_stats[g][k] for k in stat_keys if k in shared_stats[g]}
            per_group_labels[g] = label_group[g]

        # Collect all unique labels across all groups and determine which groups have each label
        label_sets = {g: set(per_group_labels[g].flatten().tolist()) for g in group_keys}
        all_labels = set()
        for s in label_sets.values():
            all_labels |= s

        # For each label, compute PoE across groups that have cells with that label
        poe_stats_per_label = {}
        for label in all_labels:
            groups_with_label = [g for g in group_keys if label in label_sets[g]]
            groups_without_label = [g for g in group_keys if label not in label_sets[g]]

            # Build stats dict for _poe_n: groups with label contribute real stats,
            # groups without contribute uninformative prior
            label_stats_for_poe = {}
            for g in groups_with_label:
                mask = (per_group_labels[g] == label).squeeze()
                label_stats_for_poe[g] = {key: value[mask] for key, value in per_group_stats[g].items()}

            if len(groups_with_label) >= 2:
                # Enough groups for PoE — also add uninformative priors for missing groups
                for g in groups_without_label:
                    ref_g = groups_with_label[0]
                    n_cells = label_stats_for_poe[ref_g]["logtheta_loc"].shape[0]
                    latent_dim = label_stats_for_poe[ref_g]["logtheta_loc"].shape[1]
                    device = label_stats_for_poe[ref_g]["logtheta_loc"].device
                    label_stats_for_poe[g] = {
                        "logtheta_loc": torch.zeros(n_cells, latent_dim, device=device),
                        "logtheta_logvar": torch.zeros(n_cells, latent_dim, device=device),
                        "logtheta_scale": torch.ones(n_cells, latent_dim, device=device),
                    }

                poe_result = self._poe_n(label_stats_for_poe)

                # For groups without this label, replace result with empty tensors
                for g in groups_without_label:
                    latent_dim = poe_result[groups_with_label[0]]["logtheta_loc"].shape[1]
                    device = poe_result[groups_with_label[0]]["logtheta_loc"].device
                    poe_result[g] = {
                        k: torch.empty((0, latent_dim), device=device) for k in stat_keys
                    }

                poe_stats_per_label[label] = poe_result
            else:
                # Only one group has this label — combine with uninformative prior
                g_with = groups_with_label[0]
                real_stats = label_stats_for_poe[g_with]
                n_cells = real_stats["logtheta_loc"].shape[0]
                latent_dim = real_stats["logtheta_loc"].shape[1]
                device = real_stats["logtheta_loc"].device

                # Create a dummy second group for the PoE (uninformative prior)
                dummy_key = -1  # temporary key not in group_keys
                dummy_stats = {
                    "logtheta_loc": torch.zeros(n_cells, latent_dim, device=device),
                    "logtheta_logvar": torch.zeros(n_cells, latent_dim, device=device),
                    "logtheta_scale": torch.ones(n_cells, latent_dim, device=device),
                }
                poe_result = self._poe_n({g_with: real_stats, dummy_key: dummy_stats})

                # Build final result: only the group with the label gets real stats
                final_result = {}
                for g in group_keys:
                    if g == g_with:
                        final_result[g] = poe_result[g_with]
                    else:
                        final_result[g] = {k: torch.empty((0, latent_dim), device=device) for k in stat_keys}
                poe_stats_per_label[label] = final_result

        # Reassemble: for each group, fill output tensors in original cell order
        # Determine device from first group's stats
        ref_device = per_group_stats[group_keys[0]]["logtheta_loc"].device
        latent_dim = per_group_stats[group_keys[0]]["logtheta_loc"].shape[1]

        concat_poe_stats = {}
        for g in group_keys:
            n_cells = per_group_stats[g]["logtheta_loc"].shape[0]
            group_output = {k: torch.empty(n_cells, latent_dim, dtype=torch.float32, device=ref_device) for k in stat_keys}

            label_count = {}
            for i, label_tensor in enumerate(per_group_labels[g]):
                label = label_tensor.item()
                count = label_count.get(label, 0)
                label_count[label] = count + 1
                poe_g_stats = poe_stats_per_label[label][g]
                tensor_index = count % poe_g_stats["logtheta_loc"].size(0)
                for k in stat_keys:
                    group_output[k][i] = poe_g_stats[k][tensor_index]

            concat_poe_stats[g] = group_output

        # Compute qz, log_z, theta for each group
        for g in group_keys:
            concat_poe_stats[g]["logtheta_qz"] = Normal(
                concat_poe_stats[g]["logtheta_loc"], concat_poe_stats[g]["logtheta_scale"].clamp(min=1e-6)
            )
            concat_poe_stats[g]["logtheta_log_z"] = concat_poe_stats[g]["logtheta_qz"].rsample()
            concat_poe_stats[g]["logtheta_theta"] = F.softmax(concat_poe_stats[g]["logtheta_log_z"], -1)

        return concat_poe_stats

    @auto_move_data
    def generative(self, private_stats, shared_stats, poe_stats, library, groups, batch_index, **kwargs):
        """Runs the generative model."""
        if self.is_multimodal:
            return self._generative_multimodal(
                private_stats, shared_stats, poe_stats, library, groups, batch_index, **kwargs
            )

        n_groups = len(private_stats)

        # Concatenate private and PoE latents for each group
        private_poe = {}
        for group in range(n_groups):
            private_log_z = private_stats[group]["log_z"]
            private_theta = private_stats[group]["theta"]
            poe_log_z = poe_stats[group]["logtheta_log_z"]
            poe_theta = poe_stats[group]["logtheta_theta"]
            private_poe[group] = {
                "log_z": torch.cat((private_log_z, poe_log_z), dim=-1),
                "theta": torch.cat((private_theta, poe_theta), dim=-1),
            }

        shared_stats_out = {}

        poe_stats_out = {}
        for (group, stats), batch in zip(private_poe.items(), batch_index):
            key = str(group)
            decoder = self.decoders[group]
            px_scale_private, px_scale_shared, px_rate_private, px_rate_shared, px_mixing, px_scale = decoder(
                self.dispersion,
                stats["log_z"][:, self.n_dimensions_shared : self.n_dimensions_private + self.n_dimensions_shared],
                stats["log_z"][:, : self.n_dimensions_shared],
                library[group],
                batch,
            )
            px_r = torch.exp(self.px_r[group])
            px = NegativeBinomialMixture(mu1=px_rate_private, mu2=px_rate_shared, theta1=px_r, mixture_logits=px_mixing)
            pz = Normal(torch.zeros_like(stats["log_z"]), torch.ones_like(stats["log_z"]))
            poe_stats_out[key] = {
                "px_scale_private": px_scale_private,
                "px_scale_shared": px_scale_shared,
                "px_rate_private": px_rate_private,
                "px_rate_shared": px_rate_shared,
                "px": px,
                "pz": pz,
            }

        outputs = {"private_shared": shared_stats_out, "private_poe": poe_stats_out}
        return outputs

    def _generative_multimodal(self, private_stats, shared_stats, poe_stats, library, groups, batch_index, **kwargs):
        """Multimodal generative model: per-(group, modality) decoding."""
        from spVIPES.module.utils import build_likelihood

        per_modality_private = kwargs.get("per_modality_private", {})
        n_groups = len(poe_stats)

        poe_stats_out = {}
        for group in range(n_groups):
            batch = batch_index[group]
            poe_log_z = poe_stats[group]["logtheta_log_z"]
            poe_theta = poe_stats[group]["logtheta_theta"]

            for modality in self.group_modalities[group]:
                # Get modality-specific private latent
                if (group, modality) in per_modality_private:
                    mod_private = per_modality_private[(group, modality)]
                else:
                    mod_private = private_stats[group]

                private_log_z = mod_private["log_z"]
                private_theta = mod_private["theta"]

                # Concatenate private + shared PoE
                combined_log_z = torch.cat((private_log_z, poe_log_z), dim=-1)

                decoder = self.decoders[(group, modality)]
                mod_library = library.get((group, modality), library.get(group))

                px_scale_private, px_scale_shared, px_rate_private, px_rate_shared, px_mixing, px_scale = decoder(
                    self.dispersion,
                    combined_log_z[:, self.n_dimensions_shared : self.n_dimensions_private + self.n_dimensions_shared],
                    combined_log_z[:, : self.n_dimensions_shared],
                    mod_library,
                    batch,
                )

                px_r_key = f"{group}_{modality}"
                px_r = torch.exp(self.px_r[px_r_key])
                likelihood_type = self.modality_likelihoods.get(modality, "nb")
                px = build_likelihood(likelihood_type, px_rate_private, px_rate_shared, px_r, px_mixing, px_scale)
                pz = Normal(torch.zeros_like(combined_log_z), torch.ones_like(combined_log_z))

                key = f"{group}_{modality}"
                poe_stats_out[key] = {
                    "px_scale_private": px_scale_private,
                    "px_scale_shared": px_scale_shared,
                    "px_rate_private": px_rate_private,
                    "px_rate_shared": px_rate_shared,
                    "px": px,
                    "pz": pz,
                }

        outputs = {"private_shared": {}, "private_poe": poe_stats_out}
        return outputs

    @torch.inference_mode()
    def get_loadings(self, dataset: int, type_latent: str) -> np.ndarray:
        """Extract per-gene weights (for each Z, shape is genes by dim(Z)) in the linear decoder."""
        # This is BW, where B is diag(b) batch norm, W is weight matrix
        self.use_batch_norm = True  # REMOVE LATER
        if type_latent not in ["shared", "private"]:
            raise ValueError(f"Invalid value for type_latent: {type_latent}. It can only be 'shared' or 'private'")
        if self.use_batch_norm is True:
            w = (
                self.decoders[dataset].factor_regressor_private.fc_layers[0][0].weight
                if type_latent == "private"
                else self.decoders[dataset].factor_regressor_shared.fc_layers[0][0].weight
            )
            bn = (
                self.decoders[dataset].factor_regressor_private.fc_layers[0][1]
                if type_latent == "private"
                else self.decoders[dataset].factor_regressor_shared.fc_layers[0][1]
            )
            sigma = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            b = gamma / sigma
            b_identity = torch.diag(b)
            loadings = torch.matmul(b_identity, w)
        else:
            loadings = (
                self.decoders[dataset].factor_regressor_private.fc_layers[0][0].weight
                if type_latent == "private"
                else self.decoders[dataset].factor_regressor_shared.fc_layers[0][0].weight
            )

        loadings = loadings.detach().cpu().numpy()
        if self.n_batch > 1:
            loadings = loadings[:, : -self.n_batch]

        return loadings

    def _compute_disentangle_losses(self, inference_outputs, tensors_by_group, n_groups, extra_metrics):
        """Compute disentanglement and contrastive losses (sum of all enabled, weighted terms).

        Each component is independently controlled by its weight at construction
        time — set the corresponding ``disentangle_*_weight`` to 0 to ablate that
        component (the network is then never created, no forward/backward cost).

        Returns
        -------
        torch.Tensor or float
            Scalar tensor sum of all enabled, weighted disentanglement terms.
            Returns the literal ``0.0`` when nothing is enabled, so the caller
            can write
            ``total_loss = total_loss + self._compute_disentangle_losses(...)``
            unconditionally.
        """
        # Quick exit when no disentanglement component is enabled
        enabled = (
            self.q_group_shared, self.q_label_shared,
            self.q_group_private, self.q_label_private, self.prototypes,
        )
        if all(x is None for x in enabled):
            return 0.0

        # Labels are needed only by the label-using components
        needs_labels = (
            self.q_label_shared is not None
            or self.q_label_private is not None
            or self.prototypes is not None
        )
        labels_by_group = None
        if needs_labels:
            labels_by_group = {
                int(k): grp["labels"].flatten()
                for grp in tensors_by_group
                for k in np.unique(grp["groups"].cpu())
            }

        disentangle_total = 0.0

        # Component 1 (q_group_shared): adversarial group erasure on z_shared
        if self.q_group_shared is not None:
            loss_val = sum(
                F.cross_entropy(
                    self.q_group_shared(gradient_reversal(
                        inference_outputs["poe_stats"][g]["logtheta_log_z"]
                    )),
                    torch.full(
                        (inference_outputs["poe_stats"][g]["logtheta_log_z"].size(0),),
                        g, dtype=torch.long,
                        device=inference_outputs["poe_stats"][g]["logtheta_log_z"].device,
                    ),
                )
                for g in range(n_groups)
            )
            disentangle_total = disentangle_total + self.disentangle_group_shared_weight * loss_val
            extra_metrics["disentangle_group_shared_loss"] = loss_val / n_groups

        # Component 2 (q_label_shared): supervised label preservation on z_shared
        if self.q_label_shared is not None:
            loss_val = sum(
                F.cross_entropy(
                    self.q_label_shared(inference_outputs["poe_stats"][g]["logtheta_log_z"]),
                    labels_by_group[g].long(),
                )
                for g in range(n_groups)
            )
            disentangle_total = disentangle_total + self.disentangle_label_shared_weight * loss_val
            extra_metrics["disentangle_label_shared_loss"] = loss_val / n_groups

        # Component 3 (q_group_private): supervised group preservation on z_private
        if self.q_group_private is not None:
            loss_val = sum(
                F.cross_entropy(
                    self.q_group_private(inference_outputs["private_stats"][g]["log_z"]),
                    torch.full(
                        (inference_outputs["private_stats"][g]["log_z"].size(0),),
                        g, dtype=torch.long,
                        device=inference_outputs["private_stats"][g]["log_z"].device,
                    ),
                )
                for g in range(n_groups)
            )
            disentangle_total = disentangle_total + self.disentangle_group_private_weight * loss_val
            extra_metrics["disentangle_group_private_loss"] = loss_val / n_groups

        # Component 4 (q_label_private): adversarial label erasure on z_private
        if self.q_label_private is not None:
            loss_val = sum(
                F.cross_entropy(
                    self.q_label_private(gradient_reversal(
                        inference_outputs["private_stats"][g]["log_z"]
                    )),
                    labels_by_group[g].long(),
                )
                for g in range(n_groups)
            )
            disentangle_total = disentangle_total + self.disentangle_label_private_weight * loss_val
            extra_metrics["disentangle_label_private_loss"] = loss_val / n_groups

        # Component 5 (contrastive): InfoNCE on z_shared via EMA prototypes
        if self.prototypes is not None:
            with torch.no_grad():
                for g in range(n_groups):
                    z = inference_outputs["poe_stats"][g]["logtheta_log_z"].detach()
                    for lbl in labels_by_group[g].unique():
                        mask = labels_by_group[g] == lbl
                        if mask.sum() > 0:
                            self.prototypes[g, lbl] = (
                                self.prototype_momentum * self.prototypes[g, lbl]
                                + (1 - self.prototype_momentum) * z[mask].mean(0)
                            )
            if n_groups > 1:
                other_groups = [
                    [gg for gg in range(n_groups) if gg != g] for g in range(n_groups)
                ]
                ct_loss = sum(
                    F.cross_entropy(
                        F.normalize(inference_outputs["poe_stats"][g]["logtheta_log_z"], dim=-1)
                        @ F.normalize(self.prototypes[other_groups[g]].mean(0), dim=-1).T
                        / self.contrastive_temperature,
                        labels_by_group[g].long(),
                    )
                    for g in range(n_groups)
                )
                disentangle_total = disentangle_total + self.contrastive_weight * ct_loss
                extra_metrics["contrastive_loss"] = ct_loss / n_groups

        return disentangle_total

    def loss(
        self,
        tensors_by_group,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        """Loss function with optional NF prior KL and cycle consistency."""
        if self.is_multimodal:
            return self._loss_multimodal(tensors_by_group, inference_outputs, generative_outputs, kl_weight)

        x = {int(k): group[REGISTRY_KEYS.X_KEY] for group in tensors_by_group for k in np.unique(group["groups"].cpu())}
        x = {i: xs[:, self.groups_var_indices[i]] for i, xs in x.items()}

        if self.log_variational_generative:
            x = {i: torch.log(1 + xs) for i, xs in x.items()}  # logvariational

        n_groups = len(x)
        extra_metrics = {}
        reconst_losses = {}
        kl_local = {}
        total_loss = None

        for g in range(n_groups):
            # Reconstruction loss
            recon_loss = -generative_outputs["private_poe"][str(g)]["px"].log_prob(x[g]).sum(-1)

            # KL divergence — private latent
            qz_private = inference_outputs["private_stats"][g]["qz"]
            z_private = inference_outputs["private_stats"][g]["log_z"]
            if self.use_nf_prior and self.nf_target in ("private", "both"):
                kl_private = self._nf_kl(qz_private, z_private, "private")
            else:
                kl_private = kl(
                    qz_private,
                    Normal(torch.zeros_like(z_private), torch.ones_like(z_private)),
                ).sum(dim=1)

            # KL divergence — shared (PoE) latent
            qz_poe = inference_outputs["poe_stats"][g]["logtheta_qz"]
            z_poe = inference_outputs["poe_stats"][g]["logtheta_log_z"]
            if self.use_nf_prior and self.nf_target in ("shared", "both"):
                kl_poe = self._nf_kl(qz_poe, z_poe, "shared")
            else:
                kl_poe = kl(
                    qz_poe,
                    Normal(torch.zeros_like(z_poe), torch.ones_like(z_poe)),
                ).sum(dim=1)

            extra_metrics[f"kl_divergence_private_group_{g}"] = kl_private.mean()
            extra_metrics[f"kl_divergence_poe_group_{g}"] = kl_poe.mean()
            reconst_losses[f"reconst_loss_group_{g}_poe"] = recon_loss
            kl_local[f"kl_divergence_group_{g}_private"] = kl_private
            kl_local[f"kl_divergence_group_{g}_poe"] = kl_poe

            group_loss = recon_loss + kl_weight * kl_private + kl_weight * kl_poe
            total_loss = group_loss if total_loss is None else total_loss + group_loss

        total_loss = total_loss + self._compute_disentangle_losses(
            inference_outputs, tensors_by_group, n_groups, extra_metrics
        )

        loss = torch.mean(total_loss)

        output = LossOutput(
            loss=loss, reconstruction_loss=reconst_losses, kl_local=kl_local, extra_metrics=extra_metrics
        )

        return output

    def _loss_multimodal(self, tensors_by_group, inference_outputs, generative_outputs, kl_weight):
        """Multimodal loss: sum over groups and modalities."""
        x = {int(k): group[REGISTRY_KEYS.X_KEY] for group in tensors_by_group for k in np.unique(group["groups"].cpu())}

        n_groups = len(x)
        extra_metrics = {}
        reconst_losses = {}
        kl_local = {}
        total_loss = None

        per_modality_private = inference_outputs.get("per_modality_private", {})

        for g in range(n_groups):
            x_group = x[g]

            # Per-modality reconstruction losses
            for modality in self.group_modalities[g]:
                key = f"{g}_{modality}"
                gen_stats = generative_outputs["private_poe"][key]

                # Get modality-specific target data
                mod_var_indices = self.groups_modality_var_indices[g][modality]
                x_mod = x_group[:, mod_var_indices]

                likelihood_type = self.modality_likelihoods.get(modality, "nb")
                if likelihood_type == "nb" and self.log_variational_generative:
                    x_mod = torch.log(1 + x_mod)

                recon_loss = -gen_stats["px"].log_prob(x_mod).sum(-1)
                reconst_losses[f"reconst_loss_group_{g}_{modality}"] = recon_loss

                # Per-modality private KL (if available)
                if (g, modality) in per_modality_private:
                    qz_mod_private = per_modality_private[(g, modality)]["qz"]
                    kl_mod_private = kl(
                        qz_mod_private,
                        Normal(
                            torch.zeros_like(per_modality_private[(g, modality)]["log_z"]),
                            torch.ones_like(per_modality_private[(g, modality)]["log_z"]),
                        ),
                    ).sum(dim=1)
                    kl_local[f"kl_divergence_group_{g}_{modality}_private"] = kl_mod_private
                    extra_metrics[f"kl_divergence_private_group_{g}_{modality}"] = kl_mod_private.mean()
                else:
                    kl_mod_private = torch.zeros(x_mod.shape[0], device=x_mod.device)

                mod_loss = recon_loss + kl_weight * kl_mod_private
                total_loss = mod_loss if total_loss is None else total_loss + mod_loss

            # Per-group PoE KL (shared across modalities)
            qz_poe = inference_outputs["poe_stats"][g]["logtheta_qz"]
            kl_poe = kl(
                qz_poe,
                Normal(
                    torch.zeros_like(inference_outputs["poe_stats"][g]["logtheta_log_z"]),
                    torch.ones_like(inference_outputs["poe_stats"][g]["logtheta_log_z"]),
                ),
            ).sum(dim=1)

            extra_metrics[f"kl_divergence_poe_group_{g}"] = kl_poe.mean()
            kl_local[f"kl_divergence_group_{g}_poe"] = kl_poe
            total_loss = total_loss + kl_weight * kl_poe

        loss = torch.mean(total_loss)

        return LossOutput(
            loss=loss, reconstruction_loss=reconst_losses, kl_local=kl_local, extra_metrics=extra_metrics
        )
