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
import yaml
from websocket import create_connection
from flask import *
import os
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
    try:
        r=requests.post('http://localhost:3001/data',json=x)
    except Exception as e:
        print(e)


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
        ws.send(format('vi dummy.c\r'))
        time.sleep(2)
        ws.send(format('i'))
        time.sleep(2)
        ws.send(format('#include <stdio.h>\nvoid main(){\nprintf("Hello world");}'))
        time.sleep(2)
        ws.send(format('\x1b'))
        time.sleep(2)
        ws.send(format(':wq\r'))
        time.sleep(2)
        d=perform_action(ws,"cc dummy.c")
        if(d['status']=='0'):
            perform_action(ws,"cc dummy.c")
            ws.send(format('rm dummy.c\r'))
            return "created, ran and executed a C program! Look at your home directory"
    elif(results[0][0]=="docker"):
        ws.send(format('docker exec -it ubuntu bash\r'))
        time.sleep(3)
        x='apt update'
        d=perform_action(ws,x)
        if(d['status']=='0'):
            d=perform_action(ws,'apt install -y curl')
        ws.send(format('exit\r'))
        return "updated apt on docker and installed curl! Perform exec on the container 'ubuntu'"
    elif(results[0][0]=="sudo"):
       response=model.generate_content('Install sqlite browser on an Ubuntu machine. Once installed, dont run it. You have to return a series of Linux commands for executing the task. You must return the command in the following format; Command: some_command. When the task is completed, return Completed!')
       ws.send(format('sudo su\r'))
       time.sleep(1)
       ws.send(format('osboxes.org\r'))
       time.sleep(0.2)
       for i in range(len(response.text.split('\n'))-1):
           d=perform_action(ws,response.text.split('\n')[i].split(':')[1])
           if(d['status']!='0'):
               return "failed"
       ws.send(format('exit\r'))
       return "successfully installed sqlite browser"
       #perform_action(ws,'sudo apt update',direct=True,comm='osboxes.org\r')
       #d=perform_action(ws,'sudo apt-get install -y sqlitebrowser')
       #if(d['status']=='0'):
       #   return "successfully installed sqlite browser"
    elif(results[0][0]=="long_task"):
        ws.send(format("tmux new -s sess\r"))
        time.sleep(2)
        x=gen_id()
        ws.send(format("sleep 20"+"#;echo #"+str(x)+"#:Status=$?time-$((`date '+%s'`))_\r#"+"ws_url"))
        time.sleep(0.5)
        ws.close()
        asyncio.run(websocket_connection())
        ws.send(format("sleep 20"+"#"+str(x)+"#ws_url"))
        time.sleep(0.1)
        ws.send(format('tmux attach -t sess\r'))
        while(True):
            r,_,_=select.select([ws],[],[])
            if(r):
                try:
                    x=json.loads(ws.recv())
                    post_transaction(x)
                    ws.send(format('exit\r'))
                    return str(x)
                except:
                    pass
    elif(results[0][0]=="multiple_machines"):
        with open('config.yaml','r') as f:
            x=yaml.safe_load(f)
        hosts=[]
        async def run_multiple(x):
            for i in x['hosts']:
                hosts.append(await asyncio.gather(websocket_connection(tuple(i.values()))))
        asyncio.run(run_multiple(x))
        for i in hosts:
            id=gen_id()
            perform_action(i[0],'python3 -m venv '+str(id))
            i[0].send(format('source  '+str(id)+'/bin/activate\r'))
            x={"file_transfer":os.getcwd()+'/r.txt'}
            json_data = ast.literal_eval(json.dumps(x))
            i[0].send(json.dumps(json_data))
            perform_action(i[0],'pip install -r r.txt')
            i[0].send(format('deactivate\r'))
            i[0].close()
        return "created virtual environments on all hosts"

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


def perform_action(ws,command,direct=False,comm=None):
    x={'command':command+"#;echo #"+str(gen_id())+"#:Status=$?time-$((`date '+%s'`))_\r#"+"sessionStorage.ws_url"}
    json_data = ast.literal_eval(json.dumps(x))
    ws.send(json.dumps(json_data))
    if(direct):
        time.sleep(0.5)
        ws.send(format(comm))
    d=None
    while(True):
        r,_,_=select.select([ws],[],[])
        if(r):
            try:
                d=json.loads(ws.recv())
                break
            except:
                pass
    post_transaction(d)
    return d

@app.get('/chatbot')
def chat():
    return render_template('base.html')

    
async def websocket_connection(creds=None):
    global ws
    async with aiohttp.ClientSession() as session:
        flag=1
        if(not(creds)):
            creds=('osboxes','osboxes','osboxes.org')
            flag=0
        url='http://localhost:8888?hostname=%s&username=%s&password=%s'%creds
        async with session.post(url) as response:
            dictionary = await response.json()
            host=None
            if(flag==0):
                ws = create_connection("ws://localhost:8888/ws?id="+dictionary.get('id'))
                conn = ws
            else:
                host = create_connection("ws://localhost:8888/ws?id="+dictionary.get('id'))
                conn = host
            while(True):
                r,_,_=select.select([conn],[],[])
                if r:
                    break
            print("Connection dict:", dictionary)
            x={"visualize":"1"}
            json_data = ast.literal_eval(json.dumps(x))
            conn.send(json.dumps(json_data))
            return conn

asyncio.run(websocket_connection())
if(__name__=="__main__"):
    app.run()


