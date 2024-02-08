import nltk
import numpy as np
import tflite_runtime.interpreter as tflite
import random
import pickle
from nltk.stem.lancaster import LancasterStemmer
import google.generativeai as genai
import aiohttp
import asyncio
import json
import random
import ast
import select
import requests
import time
from websocket import create_connection
from flask import *
ws=None

nltk.download('punkt')
stemmer = LancasterStemmer()
interpreter = tflite.Interpreter(model_path='m.tflite')
interpreter.allocate_tensors()
with open('i.pkl','rb') as f:
  intents=pickle.load(f)
with open('c.pkl','rb') as f:
  classes=pickle.load(f)
with open('w.pkl','rb') as f:
  words=pickle.load(f)

genai.configure(api_key='')
model=genai.GenerativeModel('gemini-pro')

def gen_id():
    return random.randint(1,1000)+random.randint(1,1000)+random.randint(1,1000)

def post_transaction(x):
    r=requests.post('http://localhost:3001/data',json=x)

def send(x):
    ws.send(x)

def recieve():
    return ws.recv()

def format(string):
    x={'command':string}
    json_data = ast.literal_eval(json.dumps(x))
    return json.dumps(json_data)

def predict(input_data):
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  floating_model = input_details[0]['dtype']
  interpreter.set_tensor(input_details[0]['index'], np.array(input_data,dtype=floating_model))
  interpreter.invoke()
  output_data = interpreter.get_tensor(output_details[0]['index'])
  return output_data

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
  # tokenize the pattern
  sentence_words = clean_up_sentence(sentence)
  # bag of words
  bag = [0] * len(words)
  for s in sentence_words:
    for i, w in enumerate(words):
      if w == s:
        bag[i] = 1
        if show_details:
          print("found in bag: %s" % w)

  return (np.array(bag))

context = {}
ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    results = predict(np.array([bow(sentence, words)]))[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    if(results[0][0]=="compile"):
        send(format('vi dummy.c\r'))
        time.sleep(2)
        send(format('i'))
        time.sleep(2)
        send(format('#include <stdio.h>\nvoid main(){\nprintf("Hello world");}'))
        time.sleep(2)
        send(format('\x1b'))
        time.sleep(2)
        send(format(':wq\r'))
        time.sleep(2)
        d=perform_action("cc dummy.c")
        post_transaction(d)
        if(d['status']=='0'):
            send(format('./a.out\r'))
            return "created, ran and executed a C program! Look at your home directory"
    elif(results[0][0]=="docker"):
        send(format('docker exec -it ubuntu bash\r'))
        time.sleep(3)
        x='apt update'
        d=perform_action(x)
        post_transaction(d)
        if(d['status']=='0'):
            d=perform_action('apt install -y curl')
            post_transaction(d)
        send(format('exit\r'))
        return "updated apt on docker and installed curl! Perform exec on the container 'ubuntu'"
    elif(results[0][0]=="sudo"):
       response=model.generate_content('Install sqlite browser on an Ubuntu machine. Once installed, dont run it. You have to return a series of Linux commands for executing the task. You must return the command in the following format; Command: some_command. When the task is completed, return Completed!')
       send(format('sudo su\r'))
       time.sleep(1)
       send(format('osboxes.org\r'))
       time.sleep(0.2)
       for i in range(len(response.text.split('\n'))-1):
           d=perform_action(response.text.split('\n')[i].split(':')[1])
           post_transaction(d)
           if(d['status']!='0'):
               return "failed"
       send(format('exit\r'))
       return "successfully installed sqlite browser"
       #perform_action('sudo apt update',direct=True,comm='osboxes.org\r')
       #d=perform_action('sudo apt-get install -y sqlitebrowser')
       #if(d['status']=='0'):
       #   return "successfully installed sqlite browser"
    elif(results[0][0]=="long_task"):
        send(format("tmux new -s sess\r"))
        time.sleep(2)
        x=gen_id()
        send(format("sleep 20"+"#;echo #"+str(x)+"#:Status=$?time-$((`date '+%s'`))_\r#"+"ws_url"))
        time.sleep(0.5)
        ws.close()
        asyncio.run(websocket_connection())
        send(format("sleep 20"+"#"+str(x)+"#ws_url"))
        time.sleep(0.1)
        send(format('tmux attach -t sess\r'))
        while(True):
            r,_,_=select.select([ws],[],[])
            if(r):
                try:
                    x=json.loads(ws.recv())
                    post_transaction(x)
                    send(format('exit\r'))
                    return str(x)
                except:
                    pass
            
app=Flask(__name__)
@app.post('/')
def predict1():
    try:
        text=request.get_json()['message']
        reply=response(text)
        message={"answer":reply}
    except Exception as e:
        print(e)
        message={"answer":"Sorry I didnt get you"}
    return jsonify(message)


def perform_action(command,direct=False,comm=None):
    x={'command':command+"#;echo #"+str(gen_id())+"#:Status=$?time-$((`date '+%s'`))_\r#"+"sessionStorage.ws_url"}
    json_data = ast.literal_eval(json.dumps(x))
    send(json.dumps(json_data))
    if(direct):
        time.sleep(0.5)
        send(format(comm))
    d=None
    while(True):
        r,_,_=select.select([ws],[],[])
        if(r):
            try:
                d=json.loads(recieve())
                break
            except:
                pass
    return d

@app.get('/chatbot')
def chat():
    return render_template('base.html')

    
async def websocket_connection():
    global ws
    async with aiohttp.ClientSession() as session:
        creds=('osboxes','osboxes','osboxes.org')
        url='http://localhost:8888?hostname=%s&username=%s&password=%s'%creds
        async with session.post(url) as response:
            dictionary = await response.json()
            ws = create_connection("ws://localhost:8888/ws?id="+dictionary.get('id'))
            while(True):
                r,_,_=select.select([ws],[],[])
                if r:
                    break
            print("Connection dict:", dictionary)
            x={"visualize":"1"}
            json_data = ast.literal_eval(json.dumps(x))
            ws.send(json.dumps(json_data))

asyncio.run(websocket_connection())
if(__name__=="__main__"):
    app.run()


