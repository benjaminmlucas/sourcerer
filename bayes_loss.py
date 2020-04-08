import torch
import torch.nn as nn

class BayesLoss(nn.modules.module.Module):
    """
    Description:
        This is a loss function used for semi-supervised domain
        adaptation, as described in "NAME OF MY PAPER HERE".
        To use, a model should be trained on all labelled data from
        the source domain with a standard cross-entropy loss,
        prior to training on the available labelled target data using
        this loss.
    Args:
        source_model (pytorch model): a model trained on all labelled
                      source data.
        target_train_qty (int): the number of labelled instances
                                available in the target domain.
        source_train_qty (int): the number of labelled instances
                                available in the source domain.
    Attributes:
        lambda: the Bayesian regularization constant
        source_param_list: a list of the values of the parameters of
                           the model after it has been trained on the
                           source domain.
    """
    def __init__(self, source_model, target_train_qty,
                 source_train_qty):
        super().__init__()
        self.lambda_ = 2.5277e-14 * ((float(target_train_qty) /
                                      float(source_train_qty)) **
                                      -3.3333)
        print("Lambda value for regularization: ", self.lambda_)
        self.__module_tuple = (nn.Linear, nn.Conv1d, nn.Conv2d,
                              nn.BatchNorm1d, nn.BatchNorm2d)
        source_param_list = []
        for source_module in source_model.modules():
            if isinstance(source_module, self.__module_tuple):
                source_param_list.append(source_module.weight)
                if source_module.bias is not None:
                    source_param_list.append(source_module.bias)
        self.source_param_list = source_param_list


    def forward(self, input, target, current_model):
        """
        Description:
            Calculates the sum of the cross-entropy and Bayesian losses
            and returns the value as a pytorch tensor.
        Args:
            input (tensor): the raw logits resulting from the
                            forward-pass of the model
            target (tensor): the correct labels of the instances
            current_model (pytorch_model): a model trained on at least
                                           some labelled target data.
        Returns:
            Loss (tensor): the value of the Bayesian-regularized loss
                           as a pytorch tensor.
        """
        cross_ent_loss = nn.CrossEntropyLoss()
        loss = cross_ent_loss(input, target)
        return self.__add_bayes_reg_loss(loss, current_model)


    def __add_bayes_reg_loss(self, loss, current_model):
        """
        (private method) Calculates the squared difference between the
        relevent parameters (weights and biases) of the current model
        and those of the source-trained model and adds this value to
        the cross-entropy loss.
        """
        current_param_list = []
        for current_module in current_model.modules():
            if isinstance(current_module, self.__module_tuple):
                current_param_list.append(current_module.weight)
                if current_module.bias is not None:
                    current_param_list.append(current_module.bias)

        for i in range(len(self.source_param_list)):
            diff = current_param_list[i].sub(self.source_param_list[i])
            sq_diff = diff.pow(2).sum()
            sq_diff_reg = self.lambda_ * sq_diff
            loss += sq_diff_reg
        return loss


if __name__ == '__main__':
    predictions = torch.randn(3, 5, requires_grad=True)
    # print(predictions)
    actual = torch.empty(3, dtype=torch.long).random_(5)
    # print(actual)

    src = torch.load("/media/benny/SeaBenBlue/pytorch_results/TempCNN_results/CNN_semisup_s_T31TEL_t_T31TDJ_run_0_model.pth")
    tgt = torch.load("/media/benny/SeaBenBlue/pytorch_results/TempCNN_results/CNN_semisup_s_T31TEL_t_T31TDJ_run_0_pgns_4_model.pth")

    optimizer = torch.optim.Adam(tgt.parameters())

    my_bayes_loss = BayesLoss(source_model=src, target_train_qty=1200, source_train_qty=12000000)
    loss = my_bayes_loss(input=predictions, target=actual, current_model=tgt)
    print("Loss: ", loss.data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
