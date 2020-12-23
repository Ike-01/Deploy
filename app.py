from __future__ import division, print_function

import folium
from branca.element import Figure
import pickle
from flask import Flask, render_template, request, flash
from flask import Markup
import numpy as np
import pandas as pd
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import tensorflow as tf
import keras

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
# from pickle import load
from PIL import Image

def find_top_confirmed(n = 15):

    import pandas as pd
    corona_df=pd.read_csv("https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv")

    corona_df2 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv")
    corona_df2=corona_df2[['Lat','Long_','Country_Region','Population']]
    corona_df2 = corona_df2.rename(columns={'Country_Region': 'Country', 'Long_': 'Long'})
    corona_df2=corona_df2.dropna()
    corona_df2.set_index(["Country"], inplace = True)

    corona_df = corona_df.rename(columns={'Country/Region': 'Country'})
    corona_df.drop(['Province/State'],axis=1,inplace=True)

    corona_df["Active"]=corona_df["Confirmed"]-corona_df["Recovered"]-corona_df["Deaths"]
    by_country = corona_df.groupby('Country').sum()[['Confirmed', 'Deaths', 'Recovered', 'Active']]

    result = pd.merge(by_country, corona_df2, on='Country')
    data = pd.merge(corona_df, corona_df2, on='Country')
    data.reset_index()
    data["Date"]=pd.to_datetime(data["Date"])

    dff = pd.DataFrame(data.Date)
    dd=pd.to_datetime(dff[('Date')]).dt.strftime('%m/%d/%Y')
    data['ConvertedDate'] = dd.values
    data.drop(['Date'],axis=1,inplace=True)

    cdf = by_country.nlargest(n, ['Confirmed'])

    return cdf, result, data

cdf, result, data =find_top_confirmed(10)
pairs=[(i,j,k,l,m) for i,j,k,l,m in zip(cdf.index, cdf['Confirmed'],cdf['Deaths'],cdf['Recovered'], cdf['Active'])]

tc=sum(result.Confirmed)
td=sum(result.Deaths)
tr=sum(result.Recovered)
ta=sum(result.Active)

tt=td+tr+ta
pds=round((td/tt)*100,2)
pr=round((tr/tt)*100,2)
pa=round((ta/tt)*100,2)

# for country names in dropdowList
countrynames =[[i,j,k] for i,j,k in zip(result.index.unique(),result['Lat'],result['Long']) ]

# load the model from disk
clf_beta = pickle.load(open('models/clf_beta.sav', 'rb'))
clf_gamma = pickle.load(open('models/clf_gamma.sav', 'rb'))

def get_Predictions(dfs, order , stop_X=0, stop_day=7, CountryorProvince='Morocco'):

    import numpy as np
    import datetime

    df=dfs[dfs['Country']==CountryorProvince] # for the country
    if len(df)==0:
         df=dfs[dfs['Country']=='Morocco'] # for the country

    df=df.iloc[-stop_day-2:]   #Getting the last one week data

    population=dfs['Population'].iloc[0]

    df["I"]=df['Confirmed']-df['Recovered']-df['Deaths']
    df["R"]=df['Recovered']+df['Deaths']
    df["n"] = [population] * len(df["I"])
    df["S"]=df['n']-df['I']-df['R']

    def calculate_lag_gamma(df, lag_list, R, I ,column_lag):
        for lag in lag_list:
            df[column_lag] =(df[R][1:]-df[R].shift(lag, fill_value=0))/(df[I].shift(lag, fill_value=0))
        return df  #(R[1:] - R[:-1]) / X[:-1]

    def calculate_lag_Beta(df, lag_list, n, I ,R,column_lag):
        for lag in lag_list:
            df[column_lag] =(df[n].shift(lag, fill_value=0)*(df[I][1:]-df[I].shift(lag, fill_value=0)+df[R][1:]-df[R].shift(lag, fill_value=0)))/(df[I].shift(lag, fill_value=0)*(df[n].shift(lag, fill_value=0)-df[I].shift(lag, fill_value=0)-df[R].shift(lag, fill_value=0)))
        return df ##data.n[:-1] * (data.X[1:] - data.X[:-1] + data.R[1:] - data.R[:-1]) / (data.X[:-1] * (data.n[:-1] - data.X[:-1] - data.R[:-1]))

    df = calculate_lag_gamma(df, range(1,2), 'R','I','gamma')
    df=calculate_lag_Beta(df,  range(1,2), 'n','I','R','beta')
    df[['Confirmed', 'Deaths']] = df[['Confirmed', 'Deaths']].astype('float64')

    # Fill null values given that we merged train-test datasets
    df['Confirmed'].fillna(0, inplace=True)
    df['Deaths'].fillna(0, inplace=True)
    df['Recovered'].fillna(0, inplace=True)
    df['gamma'].fillna(0, inplace=True)
    df['beta'].fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
#     print(country)

    day_count = 0

    S_predictP = (df['S'][-1:]).tolist() #Getting todays value
    X_predictP = (df['I'][-1:]).tolist() #Getting todays value
    R_predictP = (df['R'][-1:]).tolist() #Getting todays value


    predictP_beta = np.array(df['beta'][-order:]).tolist()
    predictP_gamma = np.array(df['gamma'][-order:]).tolist()

    while (X_predictP[-1] >= stop_X) and (day_count <= stop_day):

        next_beta = clf_beta.predict(np.asarray([predictP_beta[-order:]]))[0]
        next_gamma = clf_gamma.predict(np.asarray([predictP_gamma[-order:]]))[0]

        if next_beta < 0:
            next_beta = 0
        if next_gamma < 0:
            next_gamma = 0

        predictP_beta.append(next_beta)
        predictP_gamma.append(next_gamma)

        next_S = ((-predictP_beta[-1] * S_predictP[-1] *
                   X_predictP[-1]) / df['n'][-1:].tolist()[0]) + S_predictP[-1]
        next_X = ((predictP_beta[-1] * S_predictP[-1] * X_predictP[-1]) /
                  df['n'][-1:].tolist()[0]) - (predictP_gamma[-1] * X_predictP[-1]) + X_predictP[-1]
        next_R = (predictP_gamma[-1] * X_predictP[-1]) + R_predictP[-1]

        S_predictP.append(next_S)
        X_predictP.append(next_X)
        R_predictP.append(next_R)

        day_count += 1


    Date = df.ConvertedDate.values
    current = datetime.datetime.strptime(df['ConvertedDate'].iloc[-1], '%m/%d/%Y')
    while len(Date) < stop_day:
        current = current + timedelta(days=1)
        Date = np.append(Date, datetime.datetime.strftime(current, '%m/%d/%Y'))

    AC=[i for i in df.Confirmed]
    AR=[i for i in df.Recovered]
    AI=[i for i in df.I]

    Confirmed=[sum(x) for x in zip(*[X_predictP, R_predictP])]

    New2 = pd.concat((pd.DataFrame(AC), pd.DataFrame(np.rint(Confirmed)), pd.DataFrame(AR), pd.DataFrame(np.rint(R_predictP)), pd.DataFrame(AI), pd.DataFrame(np.rint(X_predictP)), pd.DataFrame(np.rint(S_predictP)), pd.DataFrame(Date)),axis =1)
    New2.columns = ['Confirmed','PConfirmed', 'Recovered', 'PRecovered', 'I', 'PI', 'S', 'Date']
    New2 = New2.set_index('Date')
    return New2

# app = Flask(__name__)

UPLOAD_FOLDER = './Web/uploads/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__,static_url_path='/uploads',
            static_folder='./Web/uploads',
            template_folder='./Web')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def funMap(df,Lat=7.9465,Long=-1.0232):
    m=folium.Map(location=[Lat,Long], control_scale=True,
                zoom_start=8)

    # fig=Figure(width=750, height=565)
    # fig=Figure(width=1235, height=565)
    # fig.add_child(m)

    def circle_maker(x):
        folium.Circle(location=[x[4],x[5]],
                     radius=float(x[0]/500),
                     color="red",
                     popup='confirmed cases:{}'.format(x[0])).add_to(m)
    df.apply(lambda x:circle_maker(x),axis=1)

    folium.raster_layers.TileLayer('Open Street Map').add_to(m)
    folium.raster_layers.TileLayer('Stamen Terrain').add_to(m)
    folium.raster_layers.TileLayer('Stamen Toner').add_to(m)
    folium.raster_layers.TileLayer('Stamen Watercolor').add_to(m)
    folium.raster_layers.TileLayer('CartoDB Positron').add_to(m)
    folium.raster_layers.TileLayer('CartoDB Dark_Matter').add_to(m)
    # add layer control to show different maps
    folium.LayerControl().add_to(m)
    html_map=m._repr_html_()
    return html_map

@app.route('/index.html')
def home():
   return render_template("index.html",table=cdf,pairs=pairs,com=tc,act=ta,rec=tr,ded=td,pa=pa,pr=pr,pds=pds,counter=countrynames)

@app.route('/')
def index():
    return render_template("index.html",table=cdf,pairs=pairs,com=tc,act=ta,rec=tr,ded=td,pa=pa,pr=pr,pds=pds,counter=countrynames)

@app.route('/predict',methods=['POST'])
def predict():
    country= request.form['country']
    days= request.form['days']

    pred=get_Predictions(data, 3, 0, int(days), CountryorProvince=country)
    results=[(i,j,k,l,m,n,x) for i,j,k,l,m,n,x in zip(pred.index, pred['Confirmed'],pred['PConfirmed'],pred['Recovered'], pred['PRecovered'], pred['I'] , pred['PI'])]

    return render_template("index.html",table=cdf,pairs=pairs,com=tc,act=ta,rec=tr,ded=td,pa=pa,pr=pr,pds=pds,results=results,counter=countrynames)

# ##################################################### for map
def countryname(df,name='Ghana'):
    data = df[df.index==name]
    Lat=data['Lat'].values
    Long=data['Long'].values
    return Lat[0],Long[0]

#pred=get_Predictions(data, 3, 0, 7)
#results=[(i,j,k,l,m,n,x) for i,j,k,l,m,n,x in zip(pred.index, pred['Confirmed'],pred['PConfirmed'],pred['Recovered'], pred['PRecovered'], pred['I'] , pred['PI'])]

#m=folium.Map(location=[7.9465,-1.0232],width=750, height=1000,,control_scale=True,tiles='Stamen toner',tiles = "Stamen Terrain",
#            zoom_start=8)


@app.route('/countrymap.html')
def countrymap():
   html_map=funMap(result)
   return render_template("countrymap.html",table=cdf,cmap=html_map,pairs=pairs,com=tc,act=ta,rec=tr,ded=td,pa=pa,pr=pr,pds=pds,counter=countrynames)

@app.route('/cmap',methods=['POST'])
def cmap():
    country=str(request.form['country'])
    Lat,Long=countryname(result,country)
    html_map=funMap(result,Lat,Long)
    return render_template("countrymap.html",table=cdf,cmap=html_map,pairs=pairs,com=tc,act=ta,rec=tr,ded=td,pa=pa,pr=pr,pds=pds,counter=countrynames)

chatbot = ChatBot(
    'CoronaBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.BestMatch',
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, but I do not understand. I am still learning.',
            'maximum_similarity_threshold': 0.90
        }
    ],
    database_uri='sqlite:///database.sqlite3'
)

# trainer = ListTrainer(chatbot)
# training_data = open('coronaNewEdit.txt').read().splitlines()
# trainer.train(training_data)
#  Training with English Corpus Data
# trainer_corpus = ChatterBotCorpusTrainer(chatbot)


@app.route("/pred.html")
def pred():
    return render_template("pred.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    result=str(chatbot.get_response(userText))
    if 'The current time is' in result:
        result="I am sorry, I don't understand that. I am still learning. Thanks"
    else:
        result=str(chatbot.get_response(userText))
    return result


@app.route("/uploadRecom.html")
def uploadRecom():
    return render_template("uploadRecom.html")

# define a function that creates similarity matrix
# if it doesn't exist
def create_sim():
    data = pd.read_csv('data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    sim = cosine_similarity(count_matrix)
    return data,sim


# defining a function that recommends 10 most similar movies
def rcmd(m):
    m = m.lower()
    # check if data and sim are already assigned
    try:
        data.head()
        sim.shape
    except:
        data, sim = create_sim()
    # check if the movie is in our database or not
    if m not in data['movie_title'].unique():
        return('This movie is not in our database.\nPlease check if you spelled it correct.')
    else:
        # getting the index of the movie in the dataframe
        i = data.loc[data['movie_title']==m].index[0]

        # fetching the row containing similarity scores of the movie
        # from similarity matrix and enumerate it
        lst = list(enumerate(sim[i]))

        # sorting this list in decreasing order based on the similarity score
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)

        # taking top 1- movie scores
        # not taking the first index since it is the same movie
        lst = lst[1:11]

        # making an empty list that will containg all 10 movie recommendations
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = rcmd(movie)
    movie = movie.upper()
    if type(r)==type('string'):
        return render_template('uploadRecom.html',movie=movie,r=r,t='s')
    else:
        return render_template('uploadRecom.html',movie=movie,r=r,t='l')


@app.route('/uploadAd.html')
def uploadAd():
   return render_template('uploadAd.html')

Adsmodel = pickle.load(open('models/Adsmodel.pkl', 'rb'))

@app.route('/Adspredict',methods=['POST'])
def Adspredict():
    tv= int(request.form['tv'])
    radio= int(request.form['radio'])
    news= int(request.form['Newspaper'])
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = Adsmodel.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('uploadAd.html', prediction_text='Spending $ {} on TV advertistment, $ {} on RADIO and $ {} on NEWSPAPER will yield an income of $ {}'.format(tv, radio, news, output))

@app.route('/uploadSalary.html')
def uploadsalary():
   return render_template('uploadSalary.html')

salarymodel = pickle.load(open('models/salarymodel.pkl', 'rb'))

@app.route('/Salarypredict',methods=['POST'])
def Salarypredict():
    experience= int(request.form['experience'])
    testscore= int(request.form['testscore'])
    interviewscore= int(request.form['interviewscore'])
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = salarymodel.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('uploadSalary.html', prediction_text='Employee having work experience of {} year(s), a test score of {} and interview score of {} will earn a salary of $ {}'.format(experience, testscore, interviewscore, output))


def model_predictDiabetes(modelsec, final_features):

    D={'LogisticRegressionCV':'diaLR.sav',
   'GaussianNB':'diaclfgnb.sav',
   'SGDClassifier':'diaclfSGD.sav',
   'LinearSVC':'diaclfSVC.sav',
   'RandomForestClassifier':'diarf.sav',
   'SVC':'diaSVM.sav',
   'KNeighborsClassifier':'diaknn.sav'}

    model=D[modelsec]

    if model=="":
        model='diaLR.sav'

    models=pickle.load(open('models/'+model, 'rb'))


    # x = [np.array(final_features)]

    classes = models.predict(final_features)
    classes=classes[0]

    pred_proba = models.predict_proba(final_features)
    result=pred_proba.argmax(axis=1)

    classe={1: 'You have Diabetes', 0:'You don\'t have Diabetes'}

    predictprob=pred_proba[0][result[0]]
    predictc=classe[classes]

    predictprob= '%.0f' % (predictprob*100)

    if predictc=='You have Diabetes':
        predictc=predictc + ' with the probability of ' + predictprob + '%.' + ' Please consult a Doctor.'
    else:
        predictc=predictc + ' with the probability of ' + predictprob + '%.'
    return predictc

@app.route('/uploadDiabetes.html')
def uploadD():
   return render_template('uploadDiabetes.html')

@app.route('/Dpredict',methods=['POST'])
def Dpredict():
    modelsec= request.form['models']
    p= int(request.form['pregnancies'])
    g= int(request.form['glucose'])
    b= int(request.form['bloodbressure'])
    s= int(request.form['skinthickness'])
    i= int(request.form['insulin'])
    dpf= int(request.form['dpf'])
    bmi= int(request.form['bmi'])
    a= int(request.form['age'])
    # final_features = [int(x) for x in request.form.values()]
    final_features=[p,g,b,s,i,bmi,dpf,a]
    int_features=[np.array(final_features)]
    predictc =model_predictDiabetes(modelsec, int_features)
    return render_template('uploadDiabetes.html', prediction_text=predictc,v=final_features)


@app.route('/CoronaFakenews.html')
def CoronaFakenews():
   return render_template('CoronaFakenews.html')

def classify(document,modelsec):
    import numpy as np
    import pickle

    from sklearn.feature_extraction.text import HashingVectorizer

    m={'RandomForestClassifier':'fake_coronaRF_model.sav',
   'SVM':'fake_coronaSVM_model.sav',
   'KNeighborsClassifier':'fake_coronaKNN_model.sav',
   'DecisionTreeClassifier':'fake_coronaDT_model.sav'}

    model=m[modelsec]
    if model=="":
      model='fake_coronaRF_model.sav'

    model = pickle.load(open('models/'+ model, 'rb'))

    vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None)

    label = {0: 'FAKE', 1: 'TRUE'}
    X = vect.transform([document])
    y = model.predict(X)[0]
    proba = np.max(model.predict_proba(X))
    proba = round(proba*100,0)
    return label[y], proba

@app.route('/predictFakeNews',methods=['POST'])
def predictFakeNews():
    models= request.form['models']
    comment= request.form['comment']
    y, proba=classify(comment,models)
    return render_template('CoronaFakenews.html',prediction=y, proba='{}%'.format(proba))


@app.route('/EmailSpam.html')
def EmailSpam():
   return render_template('EmailSpam.html')

def classifyEmail(document,modelsec):
    import numpy as np
    import pickle

    from sklearn.feature_extraction.text import HashingVectorizer

    m={'RandomForestClassifier':'emailRF_model.sav',
   'SVM':'emailSVM_model.sav',
   'KNeighborsClassifier':'emailKNN_model.sav',
   'DecisionTreeClassifier':'emailDT_model.sav'}

    model=m[modelsec]
    if model=="":
        model='emailRF_model.sav'

    model = pickle.load(open('models/'+ model, 'rb'))

    vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None)

    label = {0: 'Spam', 1: 'Ham'}
    X = vect.transform([document])
    y = model.predict(X)[0]
    proba = np.max(model.predict_proba(X))
    proba = round(proba*100,0)
    return label[y], proba

@app.route('/predictspam',methods=['POST'])
def predictspam():
    models= request.form['models']
    comment= request.form['comment']
    y, proba=classifyEmail(comment,models)
    return render_template('EmailSpam.html',prediction=y, proba='{}%'.format(proba))

#
# config = tf.ConfigProto(
#     device_count={'GPU': 1},
#     intra_op_parallelism_threads=1,
#     allow_soft_placement=True
# )
#
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.6
#
# session = tf.Session(config=config)
# keras.backend.set_session(session)

def model_predict(img_path, modelsec):

    F={'MobileNet':'MNetFlowers.h5',
    'MobileNetV2':'MNetV2Flowers.h5'}
    model=F[modelsec]

    if model=="":
        model='MNetFlowers.h5'
    model = load_model('models/'+ model)
    # model._make_predict_function()  # Necessary
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x= preprocess_input(x)
    classes = model.predict(x)
    result=classes.argmax(axis=1)
    classe={'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    key_list = list(classe.keys())
    prediction=key_list[result[0]]
    predictions=classes[0][result[0]]
    q=1-predictions
    predictions= '%.2f' % (predictions*100)
    q= '%.2f' % (q*100)
    return prediction, predictions, q

@app.route('/uploadFlowers.html')
def uploadFlowers():
   return render_template('uploadFlowers.html')

@app.route('/uploaded_flower', methods = ['POST'])
def uploaded_flower():
# with session.as_default():
#     with session.graph.as_default():
     # try:
        # check if the post request has the file part
        if request.method == 'POST':
            if request.method == 'POST':
                if 'file' not in request.files:
                    flash('No file part')
                    return redirect(request.url)
                file = request.files['file']
                if file.filename == '':
                    flash('No selected file')
                    return redirect(request.url)
                if file:
                    filename = secure_filename(file.filename)
                    file_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)

            models= request.form['models']
            prediction, predictions, q = model_predict(file_path, models)
            P='uploads/images/'+filename
            return render_template('uploadFlowers.html',p=predictions, q=q, c= prediction, F=P)
      # except Exception as e:
      #       print(e)


def model_predictCovid(img_path, modelsec):

    C={'MobileNet':'MNetcovid.h5',
   'MobileNetV2':'MNetV2covid.h5'}

    model=C[modelsec]

    if model=="":
        model='MNetcovid.h5'
    model = load_model('models/'+ model)
    # model._make_predict_function()  # Necessary
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x= preprocess_input(x)
    classes = model.predict(x)
    result=classes.argmax(axis=1)
    classe={'Covid': 0, 'Normal': 1}
    key_list = list(classe.keys())
    prediction=key_list[result[0]]
    predictions=classes[0][result[0]]
    q=1-predictions
    predictions= '%.2f' % (predictions*100)
    q= '%.2f' % (q*100)
    return prediction, predictions, q

@app.route('/upload.html')
def upload():
   return render_template('upload.html')

@app.route('/uploaded_covid', methods = ['POST'])
def uploaded_covid():
# with session.as_default():
#     with session.graph.as_default():
     # try:
        # check if the post request has the file part
        if request.method == 'POST':
            if request.method == 'POST':
                if 'file' not in request.files:
                    flash('No file part')
                    return redirect(request.url)
                file = request.files['file']
                if file.filename == '':
                    flash('No selected file')
                    return redirect(request.url)
                if file:
                    filename = secure_filename(file.filename)
                    file_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
            models= request.form['models']
            prediction, predictions, q = model_predictCovid(file_path, models)
            P='uploads/images/'+filename
            return render_template('upload.html',p=predictions, q=q, c= prediction, F=P)
      # except Exception as e:
      #       print(e)


       # https://github.com/Uttam580?tab=repositories
if __name__=="__main__":
    app.run(debug=True)
