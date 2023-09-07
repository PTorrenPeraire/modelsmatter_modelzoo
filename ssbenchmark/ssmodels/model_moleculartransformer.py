from ssbenchmark.ssmodels.base_model import ModelChoice, SSMethod
import os
import sys
import torch.nn.functional as F
import torch
# sys.path.insert(len(sys.path), os.path.abspath("/home/paula/2022/SSBenchmark/external_models/MolecularTransformer"))
# from onmt.translate import Translator, TranslationBuilder
# import onmt.inputters as inputters
# from onmt import opts
# import onmt

@ModelChoice.register_choice(call_name="forwardprediction")
class model_moleculartransformer(SSMethod):

    def __init__(self, module_path=None):
        self.model_name = "MolecularTransformer"
        if module_path is not None:
            sys.path.insert(len(sys.path), os.path.abspath(module_path))
        
    def preprocess(self, data, reaction_col):
        pass

    def process_input(self, data, reaction_col):
        pass
    
    def preprocess_store(self, data, preprocess_root, instance_name):
        pass

    def process_output(self, data_root, instance_name, k):
        pass
    
    def _get_parser(self):
        import onmt
        from onmt.utils.parse import ArgumentParser
        from onmt import opts
        parser = ArgumentParser(description='translate.py')
        opts.config_opts(parser)
        opts.translate_opts(parser)
        return parser


    def model_setup(self, use_gpu=False, **kwargs):
        parser = self._get_parser()
        if not use_gpu:
            kwargs['gpu'] = -1
        args = [f"-{k}={v}" if type(v) != bool else f"-{k}" for k, v in kwargs.items()]
        
        opt = parser.parse_args(args)
        self.opt = opt
        translator = self.build_translator(opt)
        self.model = translator
    
    def _model_call(self, X):
        scores, smiles = self.model.translate(
            src=X,
            batch_size=self.opt.batch_size
            )
        scores = [[c.item() for c in cpd] for cpd in scores]
        scores = [F.softmax(torch.Tensor(log_lhs), dim = 0) for log_lhs in scores]
        
        smiles = [["".join(c.split(" ")) for c in cpd] for cpd in smiles]
        return scores, smiles
   
   
    def smi_tokenizer(self, smi): # from MolecularTransformer
        """
        Tokenize a SMILES molecule or reaction
        """
        import re
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi)]
        assert smi == ''.join(tokens)
        return ' '.join(tokens)
        
    def model_call(self, X):
        X = [self.smi_tokenizer(x) for x in X]
        
        return self._model_call(X)
    
    
    def build_translator(self, opt, report_score=True, logger=None, out_file=None):
        from onmt.model_builder import load_test_model
        from onmt.constants import ModelTask
        from onmt.translate import GNMTGlobalScorer
        from onmt.translate.translator import Translator
                
        fields, model, model_opt = load_test_model(opt)
        scorer = GNMTGlobalScorer.from_opt(opt)
        translator = Translator.from_opt(
                model,
                fields,
                opt,
                model_opt,
                global_scorer=scorer,
                out_file=out_file,
                report_align=opt.report_align,
                report_score=report_score,
                logger=logger,
            )
        return translator
