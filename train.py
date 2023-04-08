import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import utils
from sklearn import metrics
from pytorch_pretrained_bert import BertAdam
from transformers import AlbertTokenizer, AlbertModel


def train(config,model,train_iter,dev_iter,test_iter):
    """
    模型训练方法
    """
    start_time = time.time()
    # 启动BatchNormalization 和 dropout
    model.train()
    # 拿到所有model的参数
    param_optimizer = list(model.named_parameters())
    # 不需要衰减的参数
    no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params':[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':0.01},
        {'params':[p for n,p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
    ]
    # 定好参数后写优化器
    optimizer = BertAdam(params = optimizer_grouped_parameters,
                         lr = config.learning_rate,
                         warmup=0.5,
                         t_total=len(train_iter)*config.num_epochs)
    total_batch = 0 # 记录进行多少batch
    dev_best_loss = float('inf') # 记录校验集最好的loss
    last_imporve = 0 # 记录上次校验集loss下降的batch数
    flag = False # 记录是否很久没有效果提升
    for epoch in range(config.num_epochs):
        print('Epoch{}/{}'.format(epoch+1,config.num_epochs))
        for i,(trains,labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs,labels)
            loss.backward() # 反向传播，更新梯度
            optimizer.step()
            if total_batch % 100 == 0: # 每100次输出在训练集和校验集上的效果
                true = labels.data.cpu() # 训练在Gpu上，其他减轻负担
                predit = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true,predit) # metrics在GPU上不能算
                dev_acc,dev_loss = evaluate(config,model,dev_iter)
                if dev_loss<dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(),config.save_path) # 保存模型
                    imporve = '*'
                    last_imporve = total_batch
                else:
                    imporve = ''
                time_dif = utils.get_time_dif(start_time)
                msg = 'Iter:{0:>6}, Train_loss:{1:>5.2}, Train_acc:{2:>6.2}, Val_loss:{3:>5.2}, Val_acc:{4:>6.2%}, Time:{5}{6}'
                print(msg.format(total_batch,loss.item(),train_acc,dev_loss,dev_acc, time_dif, imporve)) # 打印日志
                model.train()
            total_batch += 1
            if total_batch-last_imporve >config.require_improvement:
                # 在验证集合上loss超过 1000 batch 没有下降
                print('在校验数据集合上已经很长时间没有提升了')
                flag = True
                break
        if flag:
            break
    test(config,model,test_iter)

# 评估
def evaluate(config,model,dev_iter,test = False):

    model.eval()
    loss_total = 0
    predict_all = np.array([],dtype=int)
    label_all = np.array([],dtype=int)
    with torch.no_grad():
        for texts,labels in dev_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs,labels)
            loss_total+=loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data,1)[1].numpy()
            label_all = np.append(label_all,labels)
            predict_all = np.append(predict_all, predict)
    acc = metrics.accuracy_score(label_all,predict_all)

    if test:
        report = metrics.classification_report(label_all,predict_all,target_names=config.class_list,digits=4)
        confusion = metrics.confusion_matrix(label_all,predict_all)
        return acc, loss_total / len(dev_iter),report,confusion

    return acc, loss_total / len(dev_iter) # 输出平均loss 好对比


# 测试
def test(config,model,test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc,test_loss,test_report,test_confusion = evaluate(config, model, test_iter, test = True)
    msg = 'Test Loss:{1:>5.2}, Test Acc:{1:>6.2%}'
    print(msg.format(test_loss,test_acc))
    print('Precision,Recall,F1-score')
    print(test_report)
    print('Confusion Maxtrix')
    print(test_confusion)
    time_dif = utils.get_time_dif(start_time)
    print('使用时间：',time_dif)








