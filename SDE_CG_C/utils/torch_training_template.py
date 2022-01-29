import torch

model = ScoreNet()
#method 1
#save the model structure + parameters
torch.save(model, '/C:/user/hp/alearn/score_model.pth')
#load
model = torch.load('/C:/user/hp/alearn/score_model.pth')
#note: 此时的文件一定要能搜到模型的原始架构，否职责的话会报错

#method 2
#save a dictionary form for our model, which is only the parameter
torch.save(model.state_dict(),'/C:/user/hp/alearn/score_model.pth')
#reload the model
score_model = ScoreNet()
device = 'cuda' #'cpu'
state = torch.load('/C:/user/hp/alearn/score_model.pth', map_location=device)
score_model.load_state_dict(state)




#添加Tensorboard
writer = SummaryWriter("logs_train")

train_loader = Dataloader(train_dset,batch_size=128,shuffle=False)
val_loader = Dataloader(val_dset,batch_size=128,shuffle=False)
epochs = 10
total_train_step = 0
score_model = ScoreNet()
score_model = score_model.to(device)
total_val_step = 0
learning_rate=1e-2
optimizer = torch.optim.adam(score_model.parameters(), lr=learning_rate)
for i in range(epoch):
    print('---------第{}轮训练开始--------'.format(i+1))
    score_model.train()
    for batch in train_dataloader:
        imgs, targets = batch
        images = images.to(device)
        outputs = score_model(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()      #get the gradient
        optimizer.step()     #优化

        total_train_step = total_train_step+1
        if total_train_step % 100 == 0:
            print('train_times:{}, loss:{}'.format(total_train_step,loss))
            writer.add_scalar('train_loss', loss, total_train_step)



    score_model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            imgs, traget = batch
            outputs = score_model(imgs)
            loss = loss_fn(outputs, target)
            total_val_loss+=loss
    print(total_val_loss)
    total_val_step+=1
    writer.add_scalar("val_loss",total_val_loss, total_val_step)

    torch.save(score_model, "score_model_{}.pth".format(total_train_step))
    print('模型已保存')

writer.close()