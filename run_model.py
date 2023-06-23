import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, f1_score
from model import MSResNet
import torch.nn as nn
from utils import *



data, target_labels = get_data()
print(data.shape, len(target_labels))

model = MSResNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

train_pct = 0.77
threshold = int(train_pct * len(data))
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target_labels,
                                                    test_size=1 - train_pct,
                                                    shuffle=True,
                                                    random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train, X_test = torch.FloatTensor(X_train), torch.FloatTensor(X_test)
y_train, y_test = torch.FloatTensor(y_train), torch.FloatTensor(y_test)

train_loader = build_dataloader(X_train, y_train, batch_size=32)
test_loader = build_dataloader(X_test, y_test, batch_size=32)


def test(model_):
    # Test the model
    model_.eval()
    with torch.no_grad():
        test_predictions = []
        test_labels = []
        total = 0
        correct = 0
        num_positives = 0
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        correct_positives = []
        for _, (region, labels) in enumerate(test_loader):
            mini_batch = region.shape[0]
            region = region.reshape(-1, 1, 4, SEQLEN).to(device)
            outputs = model_(region)
            labels = labels.int().to(device)
            predicted = torch.argmax(nn.Sigmoid()(outputs), dim=1)
            total += labels.size(0)

            for i in range(labels.size(0)):
                p = predicted[i].item()
                l = labels[i].item()

                # TP
                if p == 1 and l == 1:
                    tp += 1
                # TN
                elif p == 0 and l == 0:
                    tn += 1
                # FP
                elif p == 1 and l == 0:
                    fp += 1
                # FN
                elif p == 0 and l == 1:
                    fn += 1

            for p in predicted:
                test_predictions.append(p.item())
            for l in labels:
                test_labels.append(l.item())

        print(test_predictions[:mini_batch * 2])
        print(test_labels[:mini_batch * 2])

        print(tp, tn, fp, fn)
        print("Matthews corrcoef: {}".format(matthews_corrcoef(test_labels, test_predictions)))
        print("F1 score:", f1_score(test_labels, test_predictions, average='binary'))


print("Loading pre-trained model parameters and running an evaluation on test set.")
print("This will take a few minutes...")
checkpoint = torch.load('pretrained_weights.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

print("---Test eval---")
model.eval()
test(model)


const_baseline = get_constitutive_baseline(data)
const_baseline = const_baseline.reshape(-1, 1, 4, SEQLEN).to(device)


model.train()
num_epochs = 0
total_step = len(train_loader)
for epoch in range(0, num_epochs):
    
    for i, (region, labels) in tqdm.tqdm(enumerate(train_loader), position=0, leave=True):
        # Move tensors to the configured device
        region = region.to(device)

        # Forward pass
        region = region.reshape(-1, 1, 4, SEQLEN)
        outputs = model(region)

        if (np.unique(labels.cpu()) == 0).all():
            labels = torch.FloatTensor(np.c_[np.ones(labels.shape), np.zeros(labels.shape)])
        elif (np.unique(labels.cpu()) == 1).all():
            labels = torch.FloatTensor(np.c_[np.zeros(labels.shape), np.ones(labels.shape)])
        else:
            labels = torch.FloatTensor(pd.get_dummies(labels.cpu()).values).to(device)
        labels = labels.to(device)

        loss = calculate_loss(outputs, labels).to(device)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        # Gradient Value Clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(outputs)
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.10f}'
              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        print('(Focal) loss: {:.10f}'.format(focal_loss.item()))

        test(model)

    # if (epoch + 1) % 20 == 0:
    savepath = "ResNet2D"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, savepath + "/ResNet2D{0}.pth".format(epoch + 1))


    run_deeplift(model,
                data[:num_gained_sites],
                target_labels[:num_gained_sites],
                epoch+1,
                num_gained_sites,
                const_baseline)