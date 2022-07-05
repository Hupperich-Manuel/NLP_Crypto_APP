from datetime import timedelta
from string import punctuation
#from types import NoneType
from django.urls import reverse
from django.shortcuts import get_list_or_404, render, get_object_or_404
import numpy as np
import os
import pandas as pd
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.shortcuts import render, redirect
from datetime import date, timedelta 
from gdeltdoc import GdeltDoc, Filters
from .models import Gdelt, Crypto
import torch
import psycopg2
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datetime
from django.utils import timezone
import yfinance as yf
from django.http import JsonResponse
import pickle
import joblib
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def IndexView(request):
    message=''
    return render(request, 'home/index.html', {'message':message})

def InfoExtraction(request):

    crypto_input = request.POST.get('crypto_input', False).strip().lower()
    is_json = request.POST.get('json', False)

    '''This part gets the data from gdelt'''
    words = ['cryptocurrency','crypto','cryptocurrencies','cbdc', 'ether', 'Ethereum', 'Litecoin', 'BitcoinCash', 'BitcoinSV', 'Polkadot', 'Chainlink', 'BinanceCoin', 'VeChain', 'Cosmos', 'Polkadot', 'NEO', 'Tezos', 'Tether', 'USDCoin', 'Monero', 'Dash', 'Zcash', 'Ripple', 'Cardano', 'Stellar', 'CounosX', 'bitcoin']
    words = [i.lower() for i in words]


    def scoring(x):
        score_row = []
        row = x.lower().split()
        for i in words:
            if i in row:
                score_row.append(1)
            else:
                score_row.append(0)
        return sum(score_row)


    currenday = date.today()
    today = timezone.now()

    rate = Gdelt.objects.filter(crypto=crypto_input, date__day=today.day).last()

    if rate is not None:

        delt = Gdelt.objects.filter(crypto=crypto_input, date__day=today.day)
        texts = ['Hi user!!', 'Sorry this time you were 2nd', 'therefore, you only get todays data 😉']
    
    else:
        texts = ['Hello Crypto Enthusiast.', 'Congratulations for being!', f'1st searching {crypto_input} today 🚀']
        daybefore = currenday - timedelta(days=1)
        start_date = daybefore.strftime("%Y-%m-%d")
        end_date = currenday.strftime("%Y-%m-%d")

        try:
            f = Filters(
            keyword = crypto_input,
            start_date = str(start_date),
            end_date = str(end_date),
            num_records = 5
            )

            gd = GdeltDoc()

            # Search for articles matching the filters
            articles = gd.article_search(f)
            df = articles[(articles["language"] == 'English')]
            
        except (ValueError, KeyError):
            message = 'There is no information for this crypto'
            return render(request, 'home/index.html',{'message':message})

        df['score'] = df['title'].map(lambda x: scoring(x))

        df = df.loc[df['score']>=1]
        df = df.drop('score', axis=1)


        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")



        def apply_finbert(x):
            inputs = tokenizer([x], padding = True, truncation = True, return_tensors='pt')
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=1) 
            return predictions[:, 0].tolist()[0], predictions[:, 1].tolist()[0], predictions[:, 2].tolist()[0]

        try:
            df['finbert_positive'],df['finbert_negative'],_ = zip(*df['title'].apply(lambda x: apply_finbert(x)))
        except ValueError:
            message = 'There is no information for this crypto'
            return render(request, 'home/index.html',{'message':message})

        df["finbert_max_pos_neg"] = np.where(abs(df["finbert_positive"]) > abs(df['finbert_negative']), df["finbert_positive"],-df["finbert_negative"])
        
        df["final_finbert"] =  df['finbert_max_pos_neg'].apply(lambda x: 2 if x > 0.9 else x)
        df["final_finbert"] =  df['final_finbert'].apply(lambda x: -2 if x < -0.9 else x)
 
        identify = Gdelt.objects.filter(crypto=crypto_input)

        if identify is not None:

            select_place = []
            index_id = Gdelt.objects.all().values_list('id', flat=True).order_by('-id').first()
            for index, row in enumerate(range(0, df.shape[0]), 1):

                selected_choice = Gdelt(
                    id = index_id + index,
                    date = timezone.now(),
                    crypto = crypto_input,
                    url = df.iloc[row].values[0],
                    url_mobile = df.iloc[row].values[1],
                    title = df.iloc[row].values[2],
                    seendate = df.iloc[row].values[3],
                    socialimage = df.iloc[row].values[4],
                    domain = df.iloc[row].values[5],
                    language = df.iloc[row].values[6],
                    sourcecountry = df.iloc[row].values[7],
                    finbert_positive = round(df.iloc[row].values[8],3),
                    finbert_negative = round(df.iloc[row].values[9],3),
                    final_finbert = round(df.iloc[row].values[11],3),
                    fama_french = int(list(Gdelt.objects.filter(crypto=crypto_input).values('fama_french').last().values())[0])

                )
                select_place.append(selected_choice)

            Gdelt.objects.bulk_create(select_place)

        else:

            select_place = []
            for row in range(0, df.shape[0]):

                selected_choice = Gdelt(
                    crypto = crypto_input,
                    url = df.iloc[row].values[0],
                    url_mobile = df.iloc[row].values[1],
                    title = df.iloc[row].values[2],
                    seendate = df.iloc[row].values[3],
                    socialimage = df.iloc[row].values[4],
                    domain = df.iloc[row].values[5],
                    language = df.iloc[row].values[6],
                    sourcecountry = df.iloc[row].values[7],
                    finbert_positive = round(df.iloc[row].values[8],3),
                    finbert_negative = round(df.iloc[row].values[9],3),
                    final_finbert = round(df.iloc[row].values[11],3),
                    fama_french = int(list(Gdelt.objects.filter(crypto=crypto_input).values('fama_french').last().values())[0])

                )
                select_place.append(selected_choice)
            Gdelt.objects.bulk_create(select_place)



        delt = Gdelt.objects.filter(crypto=crypto_input)

    if is_json == False:
        return render(request, 'home/results.html', {'gdelt':delt, 'texts':texts})
    else:
        delt = list(Gdelt.objects.filter(crypto=crypto_input).values())
        return JsonResponse(delt, safe = False)



def ShowTable(request):


    delt = Gdelt.objects.all()
    return render(request, 'home/results.html', {'gdelt':delt})