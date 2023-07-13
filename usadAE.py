
import torch.nn as nn
from utils import *
import torch.utils.data as data_utils
from tqdm.notebook import tqdm
import pandas as pd
import os


#device = get_default_device()
device = torch.device('cpu')
torch.manual_seed(42)


''' >> 
1.latent feature scale 문제 때문에 sigmoid써보고 학습잘 안되면 relu로 교체
2.일단 sample complexity 논문처럼 가장 단순한 bottleneck structure 시도 해보고 잘안되면 레이어 추가 
class VanillaEncoder(nn.Module):
  def __init__(self, in_size,latent_size ):
    super().__init__()
    self.linear1 = nn.Linear(in_size, int(in_size/2))
    self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
    self.linear3 = nn.Linear(int(in_size/4), latent_size)
    self.relu = nn.ReLU()
    #HE init
    #self.tanh = nn.Tanh()
    #self.sigmoid = nn.Sigmoid()


  def forward(self, x):
    out = self.relu(self.linear1(x))
    out2 = self.relu(self.linear2(out))
    z = self.relu(self.linear3(out2))
    return z

class VanillaDecoder(nn.Module):
  def __init__(self, out_size, latent_size):
    super().__init__()
    self.linear1 = nn.Linear(latent_size, int(out_size/4))
    self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
    self.linear3 = nn.Linear(int(out_size/2), out_size)
    self.relu = nn.ReLU()

  def forward(self, z):
    out = self.relu(self.linear1(z))
    out2 = self.relu(self.linear2(out))
    x_hat = self.relu(self.linear3(out2))
    return x_hat

class VanillaAutoencoder(nn.Module):
  def __init__(self, in_size,latent_size):
    super().__init__()
    self.encoder = VanillaEncoder(in_size,latent_size)
    self.decoder = VanillaDecoder(in_size,latent_size)

  def forward(self, x):
    z = self.encoder(x)
    x_hat = self.decoder(z)
    return x_hat
'''
class VanillaEncoder(nn.Module):
  def __init__(self, in_size,latent_size ):
    super().__init__()
    self.linear1 = nn.Linear(in_size, latent_size)
    #self.relu = nn.ReLU()
    self.sig = nn.Sigmoid()

  def forward(self, x):
    out = self.sig(self.linear1(x))
    return out

class VanillaDecoder(nn.Module):
  def __init__(self, out_size, latent_size):
    super().__init__()
    self.linear1 = nn.Linear(latent_size, out_size)
    #self.relu = nn.ReLU()
    self.sig = nn.Sigmoid()

  def forward(self, z):
    out = self.sig(self.linear1(z))
    return out

class VanillaAutoencoder(nn.Module):
  def __init__(self, in_size,latent_size):
    super().__init__()
    self.encoder = VanillaEncoder(in_size,latent_size)
    self.decoder = VanillaDecoder(in_size,latent_size)

  def forward(self, x):
    z = self.encoder(x)
    x_hat = self.decoder(z)
    return x_hat


class Encoder(nn.Module):
  def __init__(self, in_size, latent_size):
    super().__init__()
    self.linear1 = nn.Linear(in_size, int(in_size/2))
    self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
    self.linear3 = nn.Linear(int(in_size/4), latent_size)
    self.relu = nn.ReLU(True)

  def forward(self, w):
    out = self.linear1(w)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    z = self.relu(out)
    return z

class Decoder(nn.Module):
  def __init__(self, latent_size, out_size):
    super().__init__()
    self.linear1 = nn.Linear(latent_size, int(out_size/4))
    self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
    self.linear3 = nn.Linear(int(out_size/2), out_size)
    self.relu = nn.ReLU(True)
    self.sigmoid = nn.Sigmoid()

  def forward(self, z):
    out = self.linear1(z)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    w = self.sigmoid(out)
    return w

class UsadAE(nn.Module):
  def __init__(self, w_size, z_size, latent_size):
    super().__init__()
    self.encoder = Encoder(w_size, z_size)
    self.decoder1 = Decoder(z_size, w_size)
    self.decoder2 = Decoder(z_size, w_size)
    self.autoencoder = VanillaAutoencoder(z_size,latent_size = latent_size)
    ''' latent size = {1,2,3,4,5,...} 성능 잘 보존 하는지 성능 metric 4가지 측정'''

  def training_step(self, batch, n):
    '''1. 기존에 한 실험에서는 training step의 epoch "n"을 이어받지 않고 Loss 계산이 이어지지 않게하고 finetuning했었음'''
    '''2. epoch를 이어받아서 학습하는 것도 고려해볼만함.  '''
    '''아래는 1. 세팅 '''
    z_ = self.encoder(batch)
    z = self.autoencoder(z_)
    w1 = self.decoder1(z)
    w2 = self.decoder2(z)
    w3 = self.decoder2(self.autoencoder(self.encoder(w1)))
    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return loss1,loss2

  def validation_step(self, batch, n):
    z = self.encoder(batch)
    z_ = self.autoencoder(z)
    w1 = self.decoder1(z_)
    w2 = self.decoder2(z_)
    w3 = self.decoder2(self.autoencoder(self.encoder(w1)))
    loss1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean((batch - w3) ** 2)
    loss2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean((batch - w3) ** 2)

    return {'val_loss1': loss1, 'val_loss2': loss2}

  def validation_epoch_end(self, outputs):
    batch_losses1 = [x['val_loss1'] for x in outputs]
    epoch_loss1 = torch.stack(batch_losses1).mean()
    batch_losses2 = [x['val_loss2'] for x in outputs]
    epoch_loss2 = torch.stack(batch_losses2).mean()
    return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}

  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))


  def encode(self, batch):
      z_ = self.encoder(batch)
      z = self.autoencoder.encoder(z_)
      return z

  def forward_score(self, batch, alpha=.5, beta=.5):
      '''finetuning model의 이상치 점수'''
      z = self.encoder(batch)
      z_ = self.autoencoder(z)
      w1 = self.decoder1(z_)
      w2 = self.decoder2(self.autoencoder(self.encoder(w1)))
      score = alpha * torch.mean((batch - w1) ** 2, axis=0) + beta * torch.mean((batch - w2) ** 2, axis=0)

      return score

def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(batch,device), n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)

'''
아래 함수는 autoencoder parameter만 finetuning 


def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    AE_optimizer = opt_func(model.autoencoder.parameters())
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch = to_device(batch, device)

            # Train AE1
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss1.backward()
            AE_optimizer.step()
            AE_optimizer.zero_grad()

            # Train AE2
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss2.backward()  
            AE_optimizer.step()
            AE_optimizer.zero_grad()

        result = evaluate(model, val_loader, epoch + 1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history
'''
''''''
def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []

    optimizer1 = opt_func(list(model.encoder.parameters()) + list(model.decoder1.parameters()) + list(model.autoencoder.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters()) + list(model.decoder2.parameters()) + list(model.autoencoder.parameters()))

    for epoch in range(epochs):
        for [batch] in train_loader:
            batch = to_device(batch, device)

            # Train AE1
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()

            # Train AE2
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()

        result = evaluate(model, val_loader, epoch + 1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def testing(model, test_loader, alpha=.5, beta=.5):
    results=[]
    for [batch] in test_loader:
        #batch=to_device(batch,device)
        w1=model.decoder1(model.autoencoder(model.encoder(batch)))
        w2=model.decoder2(model.autoencoder(model.encoder(w1)))
        results.append(alpha*torch.mean((batch-w1)**2,axis=1)+beta*torch.mean((batch-w2)**2,axis=1))

    # Move results to CPU
    #results = [res.cpu() for res in results]

    return results

def testing_ori(model, test_loader, alpha=.5, beta=.5):
    results = []
    for [batch] in test_loader:
        batch = to_device(batch, device)
        w1 = model.decoder1(model.encoder(batch))
        w2 = model.decoder2(model.encoder(w1))
        results.append(alpha * torch.mean((batch - w1) ** 2, axis=1) + beta * torch.mean((batch - w2) ** 2, axis=1))
    return results



def finetune_with_deci(data, hidden_size, decimation = 1, latent_size = 1, epochs_per_tune = 20, window_size = 12, BATCH_SIZE = 7919,
                         opt_func = torch.optim.Adam):


    loss_data = []
    loss_df = pd.DataFrame(loss_data, columns=['Decimation', 'Epoch', 'Val_Loss1', 'Val_Loss2'])

    filename = f'Loss_finetuning_deci_{decimation}.csv'

    if not os.path.isfile(filename):
        loss_df.to_csv(filename, index=False)

    training_histories = {}

    ## AVERAGE POOLING
    train_data = data.rolling(decimation).mean().dropna()[::decimation]

    # Normal to Window
    windows_normal = train_data.values[
        np.arange(window_size)[None, :] + np.arange(train_data.shape[0] - window_size)[:, None]]

    windows_normal_train = windows_normal[:int(np.floor(.8 * windows_normal.shape[0]))]
    windows_normal_val = windows_normal[
                         int(np.floor(.8 * windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

    # Dimension Setting
    w_size = windows_normal.shape[1] * windows_normal.shape[2]
    z_size = windows_normal.shape[1] * hidden_size

    # MODEL
    model = UsadAE(w_size, z_size, latent_size = latent_size)

    # Assign pre-trained weights
    pretrained_weights_path = f'model_{decimation}_deci.pth'

    # Load pre-trained weights
    load_pretrained_weights(model, pretrained_weights_path)

    # Train_loader
    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0], w_size]))
    ), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Validation Loader
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0], w_size]))
    ), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Optimizer
    '''전체 업데이트'''
    optimizer = opt_func(model.parameters())

    for epoch in range(epochs_per_tune):
        for _, inputs in enumerate(train_loader):
            inputs = inputs[0].to(device)

            # Train AE1
            loss1, loss2 = model.training_step(inputs, epoch + 1)
            loss1.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Train AE2
            loss1, loss2 = model.training_step(inputs, epoch + 1)
            loss2.backward()
            optimizer.step()
            optimizer.zero_grad()


        # Evaluate the model
        result = evaluate(model, val_loader, epoch + 1)
        model.epoch_end(epoch, result)

        # Store the validation losses for this epoch in the phase history
        training_histories[epoch + 1] = {
            'val_loss1': result['val_loss1'],
            'val_loss2': result['val_loss2']

        }

    # Save model after each phase
    torch.save(model.state_dict(), f'tuned_{decimation}_deci.pth')
    print(f'Model saved after decimation {decimation}')


    return training_histories


def encode_data(data, decimation = 1, latent_size = 1):

    BATCH_SIZE = 1
    window_size = 12


    w_size = 12 * 51
    z_size = 12 * 40


    model = UsadAE(w_size=w_size, z_size=z_size, latent_size= latent_size)

    # Load the trained model for this phase
    model_path = f'tuned_{decimation}_deci.pth'
    model.load_state_dict(torch.load(model_path))

    ## AVERAGE POOLING
    train_data = data.rolling(decimation).mean().dropna()[::decimation]

    # Normal to Window
    windows_normal = train_data.values[
        np.arange(window_size)[None, :] + np.arange(train_data.shape[0] - window_size)[:, None]]

    windows_normal_train = windows_normal[:int(np.floor(.8 * windows_normal.shape[0]))]


    # Train_loader
    normal_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0], w_size]))
    ), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    latent_variables = []

    for _, inputs in enumerate(normal_loader):
        inputs = inputs[0].to(device)

        with torch.no_grad():
            encoded_data = model.encode(inputs)
            latent_variables.append(encoded_data)


    # Convert the tensor to a pandas DataFrame
    encoded_df = pd.DataFrame(torch.cat(latent_variables, dim=0).cpu().numpy())

    # Save the DataFrame to a csv file
    encoded_df.to_csv(f'encoded_data_{decimation}_deci.csv', index=False)

    return

def compute_test_results_for_phases(test_data, attack_labels, start_phase = 0 , end_phase = 23):


    BATCH_SIZE = 7919
    window_size = 12
    # Prepare a DataFrame to store test results
    test_results = pd.DataFrame(columns=['Phase', 'F1-score', 'Precision', 'Recall','AUC', 'Threshold'])

    # Attack to Window
    windows_attack = test_data.values[np.arange(window_size)[None, :] + np.arange(test_data.shape[0] - window_size)[:, None]]


    # w_size = windows_normal.shape[1] * windows_normal.shape[2]
    w_size = 12 * 51
    z_size = 12 * 40

    # MODEL
    model = UsadAE(w_size, z_size)

    # Test Loader
    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0], w_size]))
    ), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Test label
    windows_labels = []
    for i in range(len(attack_labels) - window_size):
        windows_labels.append(list(np.int_(attack_labels[i:i + window_size])))

    y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]

    for phase in tqdm(range(start_phase, end_phase)):
        # Load the trained model for this phase
        model_path = f'tuned_phase_{phase}.pth'
        model.load_state_dict(torch.load(model_path))

        # Apply the model to the test data
        test_result = testing(model, test_loader)

        # Calculate the test metrics
        y_pred = np.concatenate([torch.stack(test_result[:-1]).flatten().detach().cpu().numpy(),
                                 test_result[-1].flatten().detach().cpu().numpy()])

        f1, best_precision, best_recall, best_threshold = calculate_max_score(y_test, y_pred)
        auc = AUC(y_test, y_pred)

        # Store the results in DataFrame
        result_data = {
            'Phase': phase,
            'F1-score': f1,
            'Precision': best_precision,
            'Recall': best_recall,
            'AUC': auc,
            'Threshold': best_threshold,
            'y_pred' : y_pred
        }
        print(result_data)
        test_results = pd.concat([test_results, pd.DataFrame(result_data, index=[0])], ignore_index=True)

    # Save DataFrame to TXT file
    test_results.to_csv('test_results_tuned.txt', sep='\t', index=False)

    return test_results



