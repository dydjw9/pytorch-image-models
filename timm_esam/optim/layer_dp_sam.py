import torch
import torch.nn.functional as F
import random

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05,weight_dropout=0.,adaptive=False,nograd_cutoff=0.0,opt_dropout=0.0,temperature=100, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.args ={"nograd_cutoff":nograd_cutoff,"opt_dropout":opt_dropout,"temperature":temperature}

        defaults = dict(rho=rho,adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["adaptive"] = adaptive
        self.weight_dropout = weight_dropout
        self.paras = None


    @torch.no_grad()
    def first_step(self, zero_grad=False):
        #first order sum 
        taylor_appro = 0
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7) / (1-self.weight_dropout)
            for p in group["params"]:
                p.requires_grad = True 
                if p.grad is None: continue
                #original sam 
                # e_w = p.grad * scale.to(p)
                #asam 
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w * 0.1)  # climb to the local maximum "w + e(w)"
                # if self.state[p]:
                    # p.sub_(self.state[p]["e_w"])
                self.state[p]["e_w"] = e_w

                taylor_appro += (p.grad**2).sum()


        if zero_grad: self.zero_grad()
        return taylor_appro * scale.to(p)


    @torch.no_grad()
    def first_half(self, zero_grad=False):
        #first order sum 
        for group in self.param_groups:
            for p in group["params"]:
                if self.state[p]:
                    p.add_(self.state[p]["e_w"]*0.90)  # climb to the local maximum "w + e(w)"


    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
                self.state[p]["e_w"] = 0
                # self.state[p] = {}

                if random.random() > (1-self.weight_dropout):
                    p.requires_grad = False

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def step(self):
        inputs,targets,loss_fct,model,defined_backward,isSAM,pickmiddle = self.paras
        assert defined_backward is not None, "Sharpness Aware Minimization requires defined_backward, but it was not provided"
        args = self.args

        # assert hasattr(model,"require_backward_grad_sync")
        # assert hasattr(model,"require_forward_param_sync")
        if isSAM:
            model.eval()
            model.require_backward_grad_sync = False
            model.require_forward_param_sync = True

            cutoff = int(len(targets) * args["nograd_cutoff"])
            if cutoff != 0:
                with torch.no_grad():
                    l_before_1 = loss_fct(inputs[:cutoff],targets[:cutoff])

            l_before_2 = loss_fct(inputs[cutoff:],targets[cutoff:])
            loss = l_before_2
            l_before = torch.cat((l_before_1,l_before_2.clone().detach()),0).detach()
            predictions = None
            return_loss = loss.clone().detach()
            self.returnthings = (predictions,return_loss)
            loss = loss.mean()
            defined_backward(loss)
            self.first_step(True)


            with torch.no_grad():
                l_after = loss_fct(inputs,targets)
                phase2_coeff = (l_after-l_before)/args["temperature"]
                coeffs = F.softmax(phase2_coeff).detach()

                #codes for sorting 
                prob = 1 - args["opt_dropout"] 
                if prob >=0.99:
                    indices = range(len(targets))
                elif not pickmiddle:
                    pp = int(len(coeffs) * prob)
                    cutoff,_ = torch.topk(phase2_coeff,pp)
                    cutoff = cutoff[-1]
                    # cutoff = 0
                    #select top k% 
                    indices = [phase2_coeff > cutoff] 
                else:
                    floating = 0.1
                    pp_head = int(len(coeffs) * (prob+floating))
                    pp_tail = int(len(coeffs) * (floating))
                    cutoff_head = torch.topk(phase2_coeff,pp_head)[0][-1]
                    cutoff_tail = torch.topk(phase2_coeff,pp_tail)[0][-1]
                    # cutoff = 0
                    #select top k% 
                    indices_head = phase2_coeff > cutoff_head
                    indices_tail = phase2_coeff < cutoff_tail
                    indices = torch.logical_and(indices_head,indices_tail)
 

            # second forward-backward step
            self.first_half()

            model.require_backward_grad_sync = True
            model.require_forward_param_sync = False


            model.train()
            loss = loss_fct(inputs[indices], targets[indices])
            loss = (loss * coeffs[indices]).sum()
            defined_backward(loss)
            self.second_step(True)
        else:
            loss = loss_fct(inputs, targets)
            loss = loss.mean()
            defined_backward(loss)
            self.base_optimizer.step()
            self.zero_grad()
            predictions = None
            self.returnthings = (predictions,loss)
            

 

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        #original sam 
                        # p.grad.norm(p=2).to(shared_device)
                        #asam 
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
