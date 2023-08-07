'''
Why we are doing train_loss / len(trainloader):
The average of the batch losses will give you an estimate of the “epoch loss” during training. Since you 
are calculating the loss anyway, you could just sum it [train_loss += loss.item()] and calculate the mean 
[train_loss / len(trainloader) => sum of loss/no of loss] after the epoch finishes. This training loss is 
used to see, how well your model performs on the training dataset. Alternatively you could also plot the 
batch loss values, but this is usually not necessary and will give you a lot of outputs.

If the loss calculation for each epoch seems wierd to you, you can use the below code. Here we are adding 
all the losses[coming from each iteration] in a list, and later we are doing mean on that list to get the 
total epoch loss. model.train() puts the model into training mode.


Use of loss.item():
.item() moves the data to CPU. It converts the value into a plain python number. And plain python 
number can only live on the CPU. But you can also try using loss directly. but this is a better option. 
More importantly, if you are new to PyTorch, it might be helpful for you to know that we use loss.item() 
to maintain running loss instead of loss because PyTorch tensors store history of its values which might 
overload your GPU very soon.

loss.item() gives the loss calculated for that batch/iteration. you need to do the mean over loss of each 
batch/iteration to get the total epoch loss. train_loss += loss.item() is summing all the loss, and at 
the last we devide by the length of the batch train_loss / len(trainloader).
'''

epochs = 5
model.train()
for e in range(epochs):
    train_loss = list()
    for data, labels in tqdm(trainloader):
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        # Clear the gradients
        optimizer.zero_grad()
        # Forward Pass
        target = model(data)
        # Find the Loss
        loss = criterion(target,labels)
        # Calculate gradients 
        loss.backward()
        # Update Weights
        optimizer.step()
        # Calculate Loss
        train_loss.append(loss.item())

    print(f'Epoch {e+1} \t\t Training Loss: {torch.tensor(train_loss).mean():.2f}')


################################################################################################################

'''
Validation loop: here model.eval() puts the model into validation mode, and by doing torch.no_grad() 
we stop the calculation of gradient for validation, coz in validation we dont update our model. 
Except evary thing is same as before. `python eval_losses=[] eval_accu=[]
'''

def test(epoch):
    model.eval()

    running_loss=0
    correct=0
    total=0

    with torch.no_grad():
        for data in tqdm(testloader):
            images,labels=data[0].to(device),data[1].to(device)

            outputs=model(images)

            loss= loss_fn(outputs,labels)
            running_loss+=loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss=running_loss/len(testloader)
    accu=100.*correct/total

    eval_losses.append(test_loss)
    eval_accu.append(accu)

    print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu)) 