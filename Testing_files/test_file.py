
# Download the data from Google Drive
import gdown
url = "https://drive.google.com/drive/folders/13ijZBGIVAm-x93YpWKqIRe9alJa-O45K?usp=sharing"
gdown.download_folder(url)


# Import Libraries
import datetime
import time
import torch
import random
import re

import numpy as np
import pandas as pd
import tensorflow as tf

# from emoji import demojize
from transformers import AutoModel, AutoTokenizer,BertForSequenceClassification, AdamW, BertConfig,get_linear_schedule_with_warmup

from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import precision_recall_fscore_support, classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import sys


# This is a special cleaning for SemEval english data only

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def normalizeToken(token):
    token = token.strip()
    lowercased_token = token.lower().strip()
    # print(token)
    if token != " ":
        if token.startswith("@"):
            return "@USER"
        elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
            return "HTTPURL"
        # elif len(token) == 1:
        #     return demojize(token)
        else:
            if token == "’":
                return "'"
            elif token == "…":
                return "..."
            else:
                return token


def normalizeTweet(tweet):
    tok = TweetTokenizer()
    tokens = tok.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])
    # print(normTweet)
    normTweet = normTweet.replace("cannot ", "can not ").replace("n't ", " n't ").replace("n 't ", " n't ").replace("ca n't", "can't").replace("ai n't", "ain't")
    normTweet = normTweet.replace("'m ", " 'm ").replace("'re ", " 're ").replace("'s ", " 's ").replace("'ll "," 'll ").replace("'d ", " 'd ").replace("'ve ", " 've ")
    normTweet = normTweet.replace(" p . m .", "  p.m.").replace(" p . m ", " p.m ").replace(" a . m ."," a.m.").replace(" a . m "," a.m ")
    normTweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normTweet)
    normTweet = normTweet.lower()
    return " ".join(normTweet.split())


from sklearn.model_selection import train_test_split

# this method just for splitting 
def splitting_method(df_, name1 ,name2, test_size = 0.5):
  y = pd.DataFrame(df_, columns = ["label"])  
  X = pd.DataFrame(df_, columns = ['sentence'])

  X_train, X_test ,y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=105)

  df_t = pd.DataFrame(X_train, columns = ['sentence'])
  df_yt = pd.DataFrame(y_train, columns = ['label'])

  train_data = pd.concat([df_t, df_yt], axis=1)
  train_data.to_csv(name1+".csv", index = False,)

  df_xtest = pd.DataFrame(X_test, columns = ['sentence'])
  df_ytest = pd.DataFrame(y_test, columns = ['label'])

  test_data = pd.concat([df_xtest, df_ytest], axis=1)
  # print(test_data.isnull().sum())
  if test_size != 0.5:
        
    test_data = test_data.drop_duplicates('sentence')
    
  test_data.to_csv(name2+".csv", index = False,)

  # return train_data
  return test_data


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# Multi_data   
temp = pd.concat(map(pd.read_csv, [ 'MULTILINGUAL EMOJI PREDICTION VIA FEDERATED LEARNING IN CLEAN AND ATTACK SCENARIOS/Train_dataset/Karim__140K_es.csv','/content/MULTILINGUAL EMOJI PREDICTION VIA FEDERATED LEARNING IN CLEAN AND ATTACK SCENARIOS/Train_dataset/Karim__300K_it.csv','/content/MULTILINGUAL EMOJI PREDICTION VIA FEDERATED LEARNING IN CLEAN AND ATTACK SCENARIOS/Train_dataset/Karim__450K_fr.csv']))
temp.to_csv("training_es_it_fr.csv", index = False,)
df_3L_train = pd.read_csv('training_es_it_fr.csv', skiprows=1, names=['Tweet','Label_2','label','sentence'])



# SemEval_data
df1 = pd.read_csv('MULTILINGUAL EMOJI PREDICTION VIA FEDERATED LEARNING IN CLEAN AND ATTACK SCENARIOS/Original_SemEval_training_and_testing_data/test/us_test.text', sep='\n\n', names=['sentence'])
df2 = pd.read_csv('MULTILINGUAL EMOJI PREDICTION VIA FEDERATED LEARNING IN CLEAN AND ATTACK SCENARIOS/Original_SemEval_training_and_testing_data/test/us_test.labels', sep='\n\n', names=['label'])

df = pd.concat([df1, df2], axis=1)
df
df.to_csv("devFile.csv", index = False,)


df1 = pd.read_csv('MULTILINGUAL EMOJI PREDICTION VIA FEDERATED LEARNING IN CLEAN AND ATTACK SCENARIOS/Original_SemEval_training_and_testing_data/us/tweet_by_ID_28_1_2019__06_28_21.txt.text', sep='\n\n', names=['sentence'])
df2 = pd.read_csv('MULTILINGUAL EMOJI PREDICTION VIA FEDERATED LEARNING IN CLEAN AND ATTACK SCENARIOS/Original_SemEval_training_and_testing_data/us/tweet_by_ID_28_1_2019__06_28_21.txt.labels', sep='\n\n', names=['label'])


df = pd.concat([df1, df2], axis=1)


df = pd.concat([df, df_3L_train], axis=0)


# you can call it :)
splitting_method(df,'centralized_dataset','fedrated_dataset',test_size = 0.5)




def main(model_name):
  # CHECKPOINT
  # CHECKPOINT = "Karim-Gamal/MMiniLM-L12-finetuned-emojis-2-client-toxic-cen-2"
  CHECKPOINT = model_name

  from transformers import AutoModelForSequenceClassification

  model = AutoModelForSequenceClassification.from_pretrained(
      CHECKPOINT, 
      num_labels = 20,   
      output_attentions = False,
      output_hidden_states = False,
  )

  model.to(device)




  # DATA loader fun
  def Data_to_dataloader(File_name):

    df = pd.read_csv(File_name)

    print('Number of training sentences: {:,}\n'.format(df.shape[0]))
    # print('Number of dev sentences: {:,}\n'.format(df_dev.shape[0]))
    df['sentence']  = df.sentence.apply(normalizeTweet)
    df.dropna()
    # df_dev['sentence']  = df_dev.sentence.apply(normalizeTweet)
    # df_dev.dropna()



    # Get the lists of sentences and their labels.
    sentences = df.sentence.values
    labels = df.label.values
    # sentences_dev = df_dev.sentence.values
    # labels_dev = df_dev.label.values
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)


    input_ids = []
    # input_ids_dev = []

    for sent in sentences:
      encoded_sent = tokenizer.encode(sent)
      input_ids.append(encoded_sent)

    # for sent_dev in sentences_dev:
    #   encoded_sent_dev = tokenizer.encode(sent_dev)
    #   input_ids_dev.append(encoded_sent_dev)


    MAX_LEN = 64
    #MAX_LEN = 128
    print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
    # input_ids_dev = pad_sequences(input_ids_dev, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
    print('\nDone.')
    # Create attention masks


    attention_masks = []
    # attention_masks_dev = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    # for sent_dev in input_ids_dev:
    #     att_mask_dev = [int(token_id > 0) for token_id in sent_dev]
    #     attention_masks_dev.append(att_mask_dev)


    train_inputs = input_ids
    # validation_inputs = input_ids_dev

    train_labels = labels
    print("train_labels: ",set(train_labels))
    
    # validation_labels = labels_dev
    # print("validation_labels: ",set(validation_labels))
    
    train_masks = attention_masks
    # validation_masks = attention_masks_dev
    
    train_inputs = torch.tensor(train_inputs)
    # validation_inputs = torch.tensor(validation_inputs)
    
    train_labels = torch.tensor(train_labels)
    print("train_labels: ",train_labels)
    
    # validation_labels = torch.tensor(validation_labels)
    # print("validation_labels: ",validation_labels)
    
    train_masks = torch.tensor(train_masks)
    # validation_masks = torch.tensor(validation_masks)


    batch_size = 64
    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    # Create the DataLoader for our validation set.
    # validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    # validation_sampler = SequentialSampler(validation_data)
    # validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return train_dataloader


  test_German = pd.read_csv('MULTILINGUAL EMOJI PREDICTION VIA FEDERATED LEARNING IN CLEAN AND ATTACK SCENARIOS/Test_dataset/german_zero_shot_54K.csv', skiprows=1, names=['Tweet','Label_2','label','sentence'])
  test_German.to_csv("test_German.csv", index = False,)

  test_es = pd.read_csv('MULTILINGUAL EMOJI PREDICTION VIA FEDERATED LEARNING IN CLEAN AND ATTACK SCENARIOS/Test_dataset/test_es_29k__final.csv', skiprows=1, names=['Tweet','Label_2','label','sentence'])
  test_es.to_csv("test_es.csv", index = False,)

  test_fr = pd.read_csv('MULTILINGUAL EMOJI PREDICTION VIA FEDERATED LEARNING IN CLEAN AND ATTACK SCENARIOS/Test_dataset/test_fr_63k__final.csv', skiprows=1, names=['Tweet','Label_2','label','sentence'])
  test_fr.to_csv("test_fr.csv", index = False,)

  test_it = pd.read_csv('MULTILINGUAL EMOJI PREDICTION VIA FEDERATED LEARNING IN CLEAN AND ATTACK SCENARIOS/Test_dataset/test_it_36k__final.csv', skiprows=1, names=['Tweet','Label_2','label','sentence'])
  test_it.to_csv("test_it.csv", index = False,)


  df_t_0 = pd.read_csv('MULTILINGUAL EMOJI PREDICTION VIA FEDERATED LEARNING IN CLEAN AND ATTACK SCENARIOS/Toxic_2_4/Client_0_Toxic_Data_25_attack_1.csv', skiprows=1, names=['sentence','Label_2','label'])
  df_t_1 = pd.read_csv('MULTILINGUAL EMOJI PREDICTION VIA FEDERATED LEARNING IN CLEAN AND ATTACK SCENARIOS/Toxic_2_4/Client_1_Toxic_Data_25_attack_1.csv', skiprows=1, names=['sentence','Label_2','label'])
  df_t_2 = pd.read_csv('MULTILINGUAL EMOJI PREDICTION VIA FEDERATED LEARNING IN CLEAN AND ATTACK SCENARIOS/Toxic_2_4/Client_2_df_25_attack_1.csv', skiprows=1, names=['sentence','label'])
  df_t_3 = pd.read_csv('MULTILINGUAL EMOJI PREDICTION VIA FEDERATED LEARNING IN CLEAN AND ATTACK SCENARIOS/Toxic_2_4/Client_3_df_25_attack_1.csv', skiprows=1, names=['sentence','label'])

  df_train_fed = pd.concat([df_t_0, df_t_1, df_t_2, df_t_3])
  df_train_fed.to_csv("df_train_fed.csv", index = False,)



  # convert CSV file to dataloader
  train_dataloader = Data_to_dataloader("df_train_fed.csv")
  validation_dataloader = Data_to_dataloader('devFile.csv') 

  German_validation_dataloader = Data_to_dataloader('test_German.csv')

  es_validation_dataloader = Data_to_dataloader('test_es.csv') 
  fr_validation_dataloader = Data_to_dataloader('test_fr.csv') 
  it_validation_dataloader = Data_to_dataloader('test_it.csv') 




  # epochs
  epochs = 30

  def train_fun(epoch_i, train_dataloader, model):

    optimizer = AdamW(model.parameters(),
                      lr = 2e-5,
                      eps = 1e-8
                    )
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)


    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    # Measure how long the training epoch takes.
    t0 = time.time()
    # Reset the total loss for this epoch.
    total_loss = 0
    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    for step, batch in enumerate(train_dataloader):

        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)

        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    loss_values.append(avg_train_loss)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    name_save = 'Model_Name'+ str(epoch_i) + '.pt'  
    torch.save(model,name_save)


  def test_fun(validation_dataloader, model):
      # ========================================
      #               Validation
      # ========================================
      # After the completion of each training epoch, measure our performance on
      # our validation set.
      print("")
      print("Running Validation...")
      t0 = time.time()
      model.eval()
      eval_loss, eval_accuracy = 0, 0
      nb_eval_steps, nb_eval_examples = 0, 0

      predictions , true_labels = [], []

      for batch in validation_dataloader:
          batch = tuple(t.to(device) for t in batch)
          b_input_ids, b_input_mask, b_labels = batch
          
          with torch.no_grad():
              # The documentation for this `model` function is here: 
              # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
              outputs = model(b_input_ids, 
                              token_type_ids=None, 
                              attention_mask=b_input_mask)

          logits = outputs[0]
          logits = logits.detach().cpu().numpy()
          label_ids = b_labels.to('cpu').numpy()
          tmp_eval_accuracy = flat_accuracy(logits, label_ids)
          eval_accuracy += tmp_eval_accuracy
          nb_eval_steps += 1

          predictions.append(logits)
          true_labels.append(label_ids)

      print('    DONE.')
      print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
      print("  Validation took: {:}".format(format_time(time.time() - t0)))
      
      flat_predictions = [item for sublist in predictions for item in sublist]
      flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
      flat_true_labels = [item for sublist in true_labels for item in sublist]
      # labelsList = ['0','1','2']
      # classif_rep = classification_report(flat_true_labels, flat_predictions, target_names=labelsList)
      classif_rep = classification_report(flat_true_labels, flat_predictions, digits=5)
      print(classif_rep)
      # Write the input text to a file
      with open("output.txt", "w") as f:
        f.write(classif_rep)




  # For each epoch...
  # for epoch_i in range(0, epochs):

  # train_fun(epoch_i, train_dataloader, model)

  print('SemEval_test_>>>')
  test_fun(validation_dataloader, model)

  # print('ES_test_>>>')
  # test_fun(es_validation_dataloader, model)

  # print('FR_test_>>>')
  # test_fun(fr_validation_dataloader, model)

  # print('IT_test_>>>')
  # test_fun(it_validation_dataloader, model)

  # print('GR_test_>>>')
  # test_fun(German_validation_dataloader, model)
          
      

if __name__ == "__main__":
    # Get the command-line argument
    args = sys.argv[1:]

    # If no argument was provided, print an error message and exit
    if not args:
        print("Please Enter the Model Name")
        sys.exit(1)

    # Get the first argument as the input text
    input_text = args[0]

    # Call the main function with the input text
    main(input_text)










