# !pip install argparse

import gradio as gr
import requests
import argparse

args_dict = dict( 
    EX_LIST = [["This is wonderful!"],
                ["Nice car"],
                ["La France est la meilleure équipe du monde"],
                ["Visca Barca"],
                ["Hala Madrid"],
                ["Buongiorno"],
                # ["Auf einigen deutschen Straßen gibt es kein Radar"],
                ["Tempo soleggiato in Italia"],
                ["Bonjour"],
                ["صباح الخير"],
                ["اكل زوجتي جميل"],
               ],

    description = 'Real-time Emoji Prediction',
    article = '<head><style>@import url(https://fonts.googleapis.com/css?family=Open+Sans:400italic,600italic,700italic,800italic,400,600,700,800)<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous"> <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.7.2/font/bootstrap-icons.css"> <link rel="stylesheet" href="https://unpkg.com/bootstrap-table@1.21.2/dist/bootstrap-table.min.css">\
    .table-responsive{-sm|-md|-lg|-xl} body{ background-color: #f5f5f5; padding: 120px 0; font-family: \'Open Sans\', sans-serif; } img{ max-width:100%; } .div_table_{ position:relative; width: max-content; margin:0 auto; } .profile-card{ position:relative; width:280px; margin:0 auto; padding:40px 30px 30px; background:#fff; border: 5px solid rgba(255,255,255,.7); text-align:center; border-radius:40px; transition: all 200ms ease; } .profile-card_2{ position:relative; width:60%; // margin:0 auto; padding:40px 30px 30px; background:#fff; border: 5px solid rgba(255,255,255,.7); text-align:center; border-radius:40px; transition: all 200ms ease; } .mask-shadow{ z-index:-1 !important; width:95%; height:12px; background:#000; bottom:0; left:0; right:0; margin:0 auto; position:absolute; border-radius:4px; opacity:0; transition: all 400ms ease-in; } .mask-shadow_2{ z-index:-1 !important; width:95%; height:12px; background:#000; bottom:0; left:0; right:0; margin:0 auto; position:absolute; border-radius:4px; opacity:0; transition: all 400ms ease-in; } .profile-card:hover{ box-shadow: 0px 30px 60px -5px rgba(55,55,71,0.3); transform: translate3d(0,-5px,0); .mask-shadow{ opacity:1; box-shadow: 0px 30px 60px -5px rgba(55,55,71,0.3); position:absolute; } } .profile-card_2:hover{ box-shadow: 0px 30px 60px -5px rgba(55,55,71,0.3); transform: translate3d(0,-5px,0); .mask-shadow{ opacity:1; box-shadow: 0px 30px 60px -5px rgba(55,55,71,0.3); position:absolute; } } .profile-card header{ display:block; margin-bottom:10px; } .profile-card_2 header{ display:block; margin-bottom:10px; } .profile-card header a{ width:150px; height:150px; display:block; border-radius:100%; margin:-120px auto 0; box-shadow: 0 0 0 5px #82b541; } .profile-card_2 header a{ width:85%; height:85%; display:block; border-radius:10%; margin:-120px auto 0; box-shadow: 0 0 0 5px #82b541; } .profile-card header a img{ border-radius: 50%; width:150px; height:150px; } .profile-card_2 header a img{ border-radius: 10%; width:100%; height:100%; } .profile-card:hover header a, .profile-card header a:hover{ animation: bounceOut .4s linear; -webkit-animation: bounceOut .4s linear; } .profile-card_2:hover header a, .profile-card header a:hover{ animation: bounceOut .4s linear; -webkit-animation: bounceOut .4s linear; } .profile-card header h1{ font-size:20px; padding:20px; color:#444; text-transform:uppercase; margin-bottom:5px; } .profile-card_2 header h1{ font-size:20px; padding:20px; color:#444; text-transform:uppercase; margin-bottom:5px; } .profile-card header h2{ font-size:14px; color:#acacac; text-transform:uppercase; margin:0; } .profile-card_2 header h2{ font-size:14px; color:#acacac; text-transform:uppercase; margin:0; } /*content*/ .profile-bio{ font-size:14px; color:#a5a5a5; line-height:1.7; font-style: italic; margin-bottom:30px; } /*link social*/ .profile-social-links{ margin:0; padding:0; list-style:none; } .profile-social-links li{ display: inline-block; margin: 0 10px; } .profile-social-links li a{ width: 55px; height:55px; display:block; background:#f1f1f1; border-radius:50%; -webkit-transition: all 2.75s cubic-bezier(0,.83,.17,1); -moz-transition: all 2.75s cubic-bezier(0,.83,.17,1); -o-transition: all 2.75s cubic-bezier(0,.83,.17,1); transition: all 2.75s cubic-bezier(0,.83,.17,1); transform-style: preserve-3d; } .profile-social-links li a img{ width:35px; height:35px; margin:10px auto 0; } .profile-social-links li a:hover{ background:#ddd; transform: scale(1.2); -webkit-transform: scale(1.2); } /*animation hover effect*/ @-webkit-keyframes bounceOut { 0% { box-shadow: 0 0 0 4px #82b541; opacity: 1; } 25% { box-shadow: 0 0 0 1px #82b541; opacity: 1; } 50% { box-shadow: 0 0 0 7px #82b541; opacity: 1; } 75% { box-shadow: 0 0 0 4px #82b541; opacity: 1; } 100% { box-shadow: 0 0 0 5px #82b541; opacity: 1; } } @keyframes bounceOut { 0% { box-shadow: 0 0 0 6px #82b541; opacity: 1; } 25% { box-shadow: 0 0 0 2px #82b541; opacity: 1; } 50% { box-shadow: 0 0 0 9px #82b541; opacity: 1; } 75% { box-shadow: 0 0 0 3px #82b541; opacity: 1; } 100% { box-shadow: 0 0 0 5px #82b541; opacity: 1; } }</style></head>',
    

    )

config = argparse.Namespace(**args_dict)




list_interface = []
list_title = []

# Preprocess text (username and link placeholders)
def preprocess(text):
    text = text.lower()
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = '' if t.startswith('http') else t
        new_text.append(t)
        # print(" ".join(new_text))
    return " ".join(new_text)


# the MMiniLM model
API_URL_MMiniLM = "https://api-inference.huggingface.co/models/Karim-Gamal/MMiniLM-L12-finetuned-emojis-IID-Fed"
headers_MMiniLM = {"Authorization": "Bearer your HF read token"}


def query_MMiniLM(payload):
    response = requests.post(API_URL_MMiniLM, headers=headers_MMiniLM, json=payload)
    return response.json()

query_MMiniLM({	"inputs": 'test',})


def _method(text):
  text = preprocess(text)
  output_temp = query_MMiniLM({
    	"inputs": text,
    })
  
  # output_dict = {d['label']: d['score'] for d in output_temp[0]}

  if output_temp:
    output_dict = {d['label']: d['score'] for d in output_temp[0]}
  else:
    # handle the case where output_temp is empty
    output_dict = {}
      
  input_list = list(output_dict.items())[:3]

  output_dict = {key: value for key, value in input_list}
  
  return output_dict


# greet("sun")

interface = gr.Interface(
    
    fn = _method, 
    
    inputs=gr.Textbox(placeholder="Enter sentence here..."), 
    outputs="label",
    examples=config.EX_LIST,
    live = True,
    
    
    title = 'MiniLM Multilingual',
    
    description=config.description,
    article = '',
    
)
list_interface.append(interface)
list_title.append('MiniLM Multilingual')



# the XLM model
API_URL_XLM = "https://api-inference.huggingface.co/models/Karim-Gamal/XLM-Roberta-finetuned-emojis-IID-Fed"
headers_XLM = {"Authorization": "Bearer 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"}


def query_XLM(payload):
    response = requests.post(API_URL_XLM, headers=headers_XLM, json=payload)
    return response.json()

query_XLM({	"inputs": 'test',})



def _method(text):
  text = preprocess(text)
  output_temp = query_XLM({
    	"inputs": text,
    })
  
  # output_dict = {d['label']: d['score'] for d in output_temp[0]}

  if output_temp:
    output_dict = {d['label']: d['score'] for d in output_temp[0]}
  else:
    # handle the case where output_temp is empty
    output_dict = {}

  input_list = list(output_dict.items())[:3]

  output_dict = {key: value for key, value in input_list}
  
  return output_dict


# greet("sun")

interface = gr.Interface(
    
    fn = _method, 
    
    inputs=gr.Textbox(placeholder="Enter sentence here..."), 
    outputs="label",
    examples=config.EX_LIST,
    live = True,
    
    
    title = 'XLM Roberta Multilingual',
    
    description=config.description,
    article = '',
    
)
list_interface.append(interface)
list_title.append('XLM Roberta Multilingual')




# the bert model
API_URL_BERT = "https://api-inference.huggingface.co/models/Karim-Gamal/BERT-base-finetuned-emojis-IID-Fed"
headers_BERT = {"Authorization": "Bearer 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"}


def query_BERT(payload):
    response = requests.post(API_URL_BERT, headers=headers_BERT, json=payload)
    return response.json()

query_BERT({	"inputs": 'test',})




def _method(text):
  text = preprocess(text)
  output_temp = query_BERT({
    	"inputs": text,
    })
  
  # output_dict = {d['label']: d['score'] for d in output_temp[0]}

  if output_temp:
    output_dict = {d['label']: d['score'] for d in output_temp[0]}
  else:
    # handle the case where output_temp is empty
    output_dict = {}

  input_list = list(output_dict.items())[:3]

  output_dict = {key: value for key, value in input_list}
  
  return output_dict


# greet("sun")

interface = gr.Interface(
    
    fn = _method, 
    
    inputs=gr.Textbox(placeholder="Enter sentence here..."), 
    outputs="label",
    examples=config.EX_LIST,
    live = True,
    
    
    title = 'BERT Multilingual',
    
    description=config.description,
    article = '',
    
)
list_interface.append(interface)
list_title.append('BERT Multilingual')




# the Switch
API_URL_Switch = "https://api-inference.huggingface.co/models/Karim-Gamal/switch-base-8-finetuned-SemEval-2018-emojis-IID-Fed"
headers_Switch = {"Authorization": "Bearer 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"}


def query_Switch(payload):
    response = requests.post(API_URL_Switch, headers=headers_Switch, json=payload)
    return response.json()

query_Switch({	"inputs": 'test',})




def _method(text):
  text = preprocess(text)
  output_temp = query_Switch({
  "inputs": text,
  })

  text_to_emoji = {'red' : '❤', 'face': '😍', 'joy':'😂', 'love':'💕', 'fire':'🔥', 'smile':'😊', 'sunglasses':'😎', 'sparkle':'✨', 'blue':'💙', 'kiss':'😘', 'camera':'📷', 'USA':'🇺🇸', 'sun':'☀' , 'purple':'💜', 'blink':'😉', 'hundred':'💯', 'beam':'😁', 'tree':'🎄', 'flash':'📸', 'tongue':'😜'}

  # Extract the dictionary from the list
  d = output_temp[0]

  # Extract the text from the 'generated_text' key
  text = d['generated_text']

  # my_dict = {}
  # my_dict[str(text_to_emoji[text.split(' ')[0]])] = 0.99
  return text_to_emoji[text.split(' ')[0]]
  

# greet("sun")

interface = gr.Interface(
    
    fn = _method, 
    
    inputs=gr.Textbox(placeholder="Enter sentence here..."), 
    outputs="text",
    examples=config.EX_LIST,
    live = True,
    
    
    title = 'Switch-Base-8',
    
    description=config.description,
    article = '',
    
)
list_interface.append(interface)
list_title.append('Switch-Base-8')



import time
# delay of 40 seconds
time.sleep(40)


def _method(input_rating):

  # tokenizer = AutoTokenizer.from_pretrained(config.CHECKPOINT_BERT)
  # model_loaded = torch.load('/content/NEW_MODELS_Imbalance/Bert/g_ex3_bert_multi_fed_data_epoch_2.pt', map_location=torch.device('cpu'))

  if input_rating <=2:
    return {'🔥': 0.6, '✨': 0.3, '💯': 0.1}

  elif input_rating <= 4 and input_rating >2:
    return {'✨': 0.6, '😉': 0.3, '💯': 0.1}

  elif input_rating >4:
    return {'😍': 0.6, '💯': 0.3, '💕': 0.1}

  # return test_with_sentance(text , config.model_loaded_bert_multi_NONIID , config.tokenizer_bert)

# greet("sun")

interface = gr.Interface(
    
    fn = _method, 
    
    inputs=gr.Slider(1, 5, value=4),
    outputs="label",
    # examples=config.EX_LIST,
    live = True,
    
    
    title = 'About us',
    
    description='We don\'t have sad emoji so our rating will always be great. 😂',
    
    # CSS Source : https://codepen.io/bibiangel199/pen/warevP

    article = config.article + '<!-- this is the markup. you can change the details (your own name, your own avatar etc.) but don’t change the basic structure! --> <div class="div_table_"> <table class="table"> <tr> <td><aside class="profile-card"> <div class="mask-shadow"></div> <header> <!-- here’s the avatar --> <a href="https://www.linkedin.com/in/hossam-amer-23b9329b/"> <img src="https://drive.google.com/uc?export=view&id=1-C_UIimeqbofJC_lldC7IQzIOX_OYRSn"> </a> <!-- the username --> <h1 style = " font-size:20px; padding:20px; color:#444;  margin-bottom:5px; " >Dr. Hossam Amer</h1> <!-- and role or location --> <h2 style = "  font-size:14px; color:#acacac; text- margin:0; " >Research Scientist at Microsoft</h2> </header> </aside></td> <td><aside class="profile-card"> <div class="mask-shadow"></div> <header> <!-- here’s the avatar --> <a href="https://www.linkedin.com/in/yuanzhu-chen-3408265/"> <img src="https://drive.google.com/uc?export=view&id=1n5Vld5FJzp1TzTISM_fB69zPHAdfDYQn"> </a> <!-- the username --> <h1 style = " font-size:20px; padding:20px; color:#444;  margin-bottom:5px; " >Dr. Yuanzhu Chen</h1> <!-- and role or location --> <h2 style = "  font-size:14px; color:#acacac; text- margin:0; ">Professor at Queen\'s University</h2> </header> </aside></td> </tr> </table> </div> <div class="div_table_"> <table class="table"> <tr> <td><aside class="profile-card"> <div class="mask-shadow"></div> <header> <!-- here’s the avatar --> <a href="https://www.linkedin.com/in/abdelrahman-elhamoly/"> <img src="https://drive.google.com/uc?export=view&id=1a9tR4Xd5XSTmx7Dy_WfRHbVCSpbgAoYb"> </a> <!-- the username --> <h1 style = " font-size:20px; padding:20px; color:#444;  margin-bottom:5px; ">Abdelrahman El-Hamoly</h1> <!-- and role or location --> <h2 style = "  font-size:14px; color:#acacac; text- margin:0; " >Master\'s student at Queen\'s University</h2> </header> </aside></td> <td><aside class="profile-card"> <div class="mask-shadow"></div> <header> <!-- here’s the avatar --> <a href="https://www.linkedin.com/in/ahmed-mohamed-gaber-143b25175/"> <img src="https://drive.google.com/uc?export=view&id=1OiGZwhL23PYhIJzQexYvPDFRrgUIprMj"> </a> <!-- the username --> <h1 style = " font-size:20px; padding:20px; color:#444;  margin-bottom:5px; ">Ahmed Gaber</h1> <!-- and role or location --> <h2 style = "  font-size:14px; color:#acacac; text- margin:0; " >Master\'s student at Queen\'s University</h2> </header> </aside></td> <td><aside class="profile-card"> <div class="mask-shadow"></div> <header> <!-- here’s the avatar --> <a href="https://www.linkedin.com/in/karim-gamal-mahmoud/"> <img src="https://drive.google.com/uc?export=view&id=1Lg2RzimITL9y__X2hycBTX10rJ4o87Ax"> </a> <!-- the username --> <h1 style=" font-size:20px; padding:20px; color:#444;  margin-bottom:5px; ">Karim Gamal</h1> <!-- and role or location --> <h2 style = "  font-size:14px; color:#acacac; text- margin:0; " >Master\'s student at Queen\'s University</h2> </header> </aside></td> </tr> </table> </div>',
    )
list_interface.append(interface)
list_title.append('About us')




demo = gr.TabbedInterface(
    list_interface, 
    list_title,
    title='Multilingual Emoji Prediction Using Federated Learning',
    css='.gradio-container {color : orange}',)
    # css='.gradio-container {background-color: white; color : orange}',)
demo.launch()

