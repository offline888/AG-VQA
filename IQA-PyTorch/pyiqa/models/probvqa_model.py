import torch
from collections import OrderedDict
from pyiqa.utils.registry import MODEL_REGISTRY
from pyiqa.models.general_iqa_model import GeneralIQAModel
from pyiqa.models.builder import build_loss 

@MODEL_REGISTRY.register()
class ProbVQAModel(GeneralIQAModel):
    def __init__(self, opt):
        super().__init__(opt)
        
        # 【新增】：从 YML 中构建客制化的 Variance Loss
        train_opt = opt['train']
        # 检查 YML 里有没有配置 var_opt
        if train_opt.get('var_opt'): 
            self.cri_var = build_loss(train_opt['var_opt']).to(self.device)
        else:
            self.cri_var = None

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            out_dict = self.net_g(self.lq)
            self.output = out_dict['quality_score'] 
        self.net_g.train()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        
        out_dict = self.net_g(self.lq)
        score = out_dict['quality_score']
        video_var = out_dict['video_var']
        text_var = out_dict['text_var']
        
        self.output = score
        
        l_total = 0
        loss_dict = OrderedDict()

        if hasattr(self, 'cri_loss') and self.cri_loss is not None:
            l_reg = self.cri_loss(score, self.gt)
            l_total += l_reg
            loss_dict['l_reg'] = l_reg
            
        if self.cri_var is not None:
            l_var = self.cri_var(video_var, text_var)
            l_total += l_var
            loss_dict['l_var'] = l_var

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)