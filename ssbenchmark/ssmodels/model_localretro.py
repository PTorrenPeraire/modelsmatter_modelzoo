"""
Wrapper for SingleStepModelZoo for LocalRetro (https://github.com/kaist-amsg/LocalRetro).

Code and logic for model setup and model call is adapted from the original source code.
Where possible the original code is used.

S. Chen and Y. Jung, “Deep Retrosynthetic Reaction Prediction using Local Reactivity and Global Attention,” 
JACS Au, vol. 1, no. 10, pp. 1612–1620, 2021, publisher: American Chemical Society. [Online]. 
Available: https://doi.org/10.1021/jacsau.1c00246
"""

import os
import sys
from functools import partial
import argparse

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dgllife.utils import smiles_to_bigraph
from rdkit import Chem
from ssbenchmark.ssmodels.base_model import Registries, SSMethod
from ssbenchmark.utils import canonicalize_smiles
from torch.utils.data import DataLoader

@Registries.ModelChoice.register(key="localretro")
class model_localretro(SSMethod):
    def __init__(self, module_path=None):
        self.model_name = "LocalRetro"
        print(module_path)
        if module_path is not None:
            sys.path.insert(len(sys.path), os.path.abspath((os.path.join(module_path, "scripts"))))
            sys.path.insert(len(sys.path), os.path.abspath((os.path.join(module_path, "LocalTemplate"))))


    def preprocess(self, data, reaction_col):
        data[reaction_col] = data[reaction_col].apply(lambda x: x.split(">"))
        data[reaction_col] = data[reaction_col].apply(
            lambda x: ">>".join([x[0], x[-1]])
        )
        return data

    def process_input(self, data, reaction_col):
        print("preprocess input")
        data = self.preprocess(data, reaction_col)
        data = data.rename(columns={reaction_col: "reactants>reagents>production"})
        return data[["reactants>reagents>production", "split"]]

    def preprocess_store(self, data, preprocess_root, instance_name):
        print("preprocess store")
        oroot = os.path.join(preprocess_root, "localretro", instance_name)
        if not self.check_path(oroot):
            self.make_path(oroot)
        for spl in ["train", "valid", "test"]:
            if spl == "valid":
                spl_ = "val"
            else:
                spl_ = spl
            d = data[data["split"] == spl][data.columns[:-1]]
            d = d.reset_index(drop=True)
            d.to_csv(os.path.join(oroot, f"raw_{spl_}.csv"))

    def process_output(self, data_root, instance_name, k):
        opath = os.path.join(
            data_root, self.model_name.lower(), f"LocalRetro_{instance_name}.txt"
        )
        if not self.check_path(opath):
            return None
        with open(
            os.path.join(
                data_root, self.model_name.lower(), f"LocalRetro_{instance_name}.txt"
            ),
            "r",
        ) as f:
            lines = f.readlines()
        data = [l.split("\t")[1 : k + 1] for l in lines]
        data = [[eval(pred)[0] for pred in prod] for prod in data]
        return [[canonicalize_smiles(p) for p in pred] for pred in data]

    def model_setup(self, use_gpu=False, **kwargs):
        from functools import partial

        from scripts.utils import init_featurizer, load_model, mkdir_p

        print("LocalRetro testing arguments")

        kwargs.setdefault("dataset", "USPTO_50K")
        kwargs.setdefault("gpu", "cuda:0")
        kwargs.setdefault("config", "default_config.json")
        kwargs.setdefault("batch_size", 16)
        kwargs.setdefault("top_num", 100)
        kwargs.setdefault("num_workers", 0)
        kwargs.setdefault("top_k", 100)
        kwargs["mode"] = "test"
        if use_gpu:
            kwargs["device"] = (
                torch.device(kwargs["gpu"])
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            kwargs["device"] = torch.device("cpu")

        print("Using device %s" % kwargs["device"])

        kwargs = init_featurizer(kwargs)
        self.model = load_model(kwargs)
        self.model.eval()
        
        atom_templates = pd.read_csv('%s/atom_templates.csv' % kwargs['template_path'])
        bond_templates = pd.read_csv('%s/bond_templates.csv' % kwargs['template_path'])
        template_infos = pd.read_csv('%s/template_infos.csv' % kwargs['template_path'])

        kwargs["rxn_class_given"] = False
        kwargs["atom_templates"] = {
            atom_templates["Class"][i]: atom_templates["Template"][i]
            for i in atom_templates.index
        }
        kwargs["bond_templates"] = {
            bond_templates["Class"][i]: bond_templates["Template"][i]
            for i in bond_templates.index
        }
        kwargs["template_infos"] = {
            template_infos["Template"][i]: {
                "edit_site": eval(template_infos["edit_site"][i]),
                "change_H": eval(template_infos["change_H"][i]),
                "change_C": eval(template_infos["change_C"][i]),
                "change_S": eval(template_infos["change_S"][i]),
            }
            for i in template_infos.index
        }
        self.args = kwargs

    def get_id_template(self, a, class_n):
        class_n = class_n  # no template
        edit_idx = a // class_n
        template = a % class_n
        return (edit_idx, template)

    def output2edit(self, out, top_num):
        class_n = out.size(-1)
        readout = out.cpu().detach().numpy()
        readout = readout.reshape(-1)
        output_rank = np.flip(np.argsort(readout))
        output_rank = [
            r for r in output_rank if self.get_id_template(r, class_n)[1] != 0
        ][:top_num]

        selected_edit = [self.get_id_template(a, class_n) for a in output_rank]
        selected_proba = [readout[a] for a in output_rank]

        return selected_edit, selected_proba

    def combined_edit(self, graph, atom_out, bond_out, top_num):
        edit_id_a, edit_proba_a = self.output2edit(atom_out, top_num)
        edit_id_b, edit_proba_b = self.output2edit(bond_out, top_num)
        edit_id_c = edit_id_a + edit_id_b
        edit_type_c = ["a"] * top_num + ["b"] * top_num
        edit_proba_c = edit_proba_a + edit_proba_b
        edit_rank_c = np.flip(np.argsort(edit_proba_c))[:top_num]
        edit_type_c = [edit_type_c[r] for r in edit_rank_c]
        edit_id_c = [edit_id_c[r] for r in edit_rank_c]
        edit_proba_c = [edit_proba_c[r] for r in edit_rank_c]

        return edit_type_c, edit_id_c, edit_proba_c

    def get_bg_partition(self, bg):
        sg = bg.remove_self_loop()
        gs = dgl.unbatch(sg)
        nodes_sep = [0]
        edges_sep = [0]
        for g in gs:
            nodes_sep.append(nodes_sep[-1] + g.num_nodes())
            edges_sep.append(edges_sep[-1] + g.num_edges())
        return gs, nodes_sep[1:], edges_sep[1:]

    def get_k_predictions(self, test_id, args):
        from template_decoder import decode_localtemplate, read_prediction

        raw_prediction = args["raw_predictions"][test_id]
        all_prediction = []
        class_prediction = []
        product = raw_prediction[0]
        predictions = raw_prediction[1:]
        for prediction in predictions:
            mol, pred_site, template, template_info, score = read_prediction(
                product,
                prediction,
                args["atom_templates"],
                args["bond_templates"],
                args["template_infos"],
            )
            local_template = ">>".join(
                ["(%s)" % smarts for smarts in template.split("_")[0].split(">>")]
            )
            decoded_smiles = decode_localtemplate(
                mol, pred_site, local_template, template_info
            )
            try:
                decoded_smiles = decode_localtemplate(
                    mol, pred_site, local_template, template_info
                )
                if (
                    decoded_smiles == None
                    or str((decoded_smiles, score)) in all_prediction
                ):
                    continue
            except Exception as e:
                continue
            all_prediction.append(str((decoded_smiles, score)))
            if args["rxn_class_given"]:
                rxn_class = args["test_rxn_class"][test_id]
                if template in args["templates_class"][str(rxn_class)].values:
                    class_prediction.append(str((decoded_smiles, score)))
                if len(class_prediction) >= args["top_k"]:
                    break

            elif len(all_prediction) >= args["top_k"]:
                break
        return (test_id, (all_prediction, class_prediction))

    def _model_call(self, X):
        from utils import collate_molgraphs_test, predict

        test_set = USPTOTestDataset(
            self.args,
            X,
            smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
            node_featurizer=self.args["node_featurizer"],
            edge_featurizer=self.args["edge_featurizer"],
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=len(X),
            collate_fn=collate_molgraphs_test,
            num_workers=self.args["num_workers"],
        )

        raw_predictions = {}
        with torch.no_grad():
            for batch_id, data in enumerate(test_loader):
                smiles_list, bg, rxns = data
                batch_atom_logits, batch_bond_logits, _ = predict(
                    self.args, self.model, bg
                )
                sg = bg.remove_self_loop()
                graphs = dgl.unbatch(sg)
                batch_atom_logits = nn.Softmax(dim=1)(batch_atom_logits)
                batch_bond_logits = nn.Softmax(dim=1)(batch_bond_logits)
                graphs, nodes_sep, edges_sep = self.get_bg_partition(bg)
                start_node = 0
                start_edge = 0
                print(
                    "\rWriting test molecule batch %s/%s"
                    % (batch_id, len(test_loader)),
                    end="",
                    flush=True,
                )
                for single_id, (graph, end_node, end_edge) in enumerate(
                    zip(graphs, nodes_sep, edges_sep)
                ):
                    smiles = smiles_list[single_id]
                    test_id = (batch_id * self.args["batch_size"]) + single_id
                    pred_types, pred_sites, pred_scores = self.combined_edit(
                        graph,
                        batch_atom_logits[start_node:end_node],
                        batch_bond_logits[start_edge:end_edge],
                        self.args["top_num"],
                    )
                    start_node = end_node
                    start_edge = end_edge
                    line = "%s\t%s\t%s\n" % (
                        test_id,
                        smiles,
                        "\t".join(
                            [
                                "(%s, %s, %s, %.3f)"
                                % (
                                    pred_types[i],
                                    pred_sites[i][0],
                                    pred_sites[i][1],
                                    pred_scores[i],
                                )
                                for i in range(self.args["top_num"])
                            ]
                        ),
                    )
                    seps = line.split("\t")
                    raw_predictions[int(seps[0])] = seps[1:]

        self.args["raw_predictions"] = raw_predictions
        results_smiles = []
        results_priors = []
        partial_func = partial(
            self.get_k_predictions, args=self.args
        )  # In source code they pool this (use multiprocessing) - could consider
        for i in range(len(raw_predictions)):
            result = partial_func(i)
            all_prediction, class_prediction = result[1]
            all_prediction = [eval(p) for p in all_prediction]
            results_smiles.append([p[0] for p in all_prediction])
            results_priors.append([p[1] for p in all_prediction])
        return results_smiles, results_priors

    def model_call(self, input):
        return self._model_call(input)


class USPTOTestDataset(object):
    def __init__(
        self,
        args,
        smiles,
        smiles_to_graph,
        node_featurizer,
        edge_featurizer,
        load=True,
        log_every=1000,
    ):
        self.smiles = smiles
        self._pre_process(smiles_to_graph, node_featurizer, edge_featurizer, load, log_every)

    def _pre_process(self, smiles_to_graph, node_featurizer, edge_featurizer, load, log_every):

        print("Processing test dgl graphs from scratch...")
        self.graphs = []
        for i, s in enumerate(self.smiles):
            if (i + 1) % log_every == 0:
                print("Processing molecule %d/%d" % (i + 1, len(self.smiles)))
            self.graphs.append(
                smiles_to_graph(
                    s,
                    node_featurizer=node_featurizer,
                    edge_featurizer=edge_featurizer,
                    canonical_atom_order=False,
                )
            )

    def __getitem__(self, item):
        return self.smiles[item], self.graphs[item], None

    def __len__(self):
        return len(self.smiles)

    # if __name__ == "__main__":


#     parser = argparse.ArgumentParser()
#     parser.add_argument("--module_path", type=str, default=None)
#     parser.add_argument("--preprocess_batch", type=str, default=None)
#     parser.add_argument("--preprocess_join", type=str, default=None)
#     parser.add_argument("--reaction_col", type=str, default="reaction_smiles")
#     parser.add_argument("--preprocess_output", type=str, default=None)
#     parser.add_argument("--instance_name", type=str, default=None)

#     args = parser.parse_args()

#     model_instance = model_chemformer(module_path=args.module_path)
#     if args.preprocess_batch is not None:
#         df = model_instance.read_csv(args.preprocess_batch)
#         model_instance.preprocess(df, args.reaction_col)
#         model_instance.write_csv(df, args.preprocess_output, str(uuid.uuid4()))

#     if args.preprocess_join is not None:
#         df = model_instance.collate_batches(args.preprocess_join)
#         model_instance.preprocess_store(df, args.preprocess_output, args.instance_name)
