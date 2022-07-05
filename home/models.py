from time import time
from django.db import models
import datetime
from django.utils import timezone


# Create your models here.

class Gdelt(models.Model):
    date = models.DateTimeField('date published',default=timezone.now)
    crypto = models.CharField('Crypto', max_length=20)
    url = models.CharField('Url', max_length=1000)
    url_mobile = models.CharField('Url mobile', max_length=1000)
    title = models.CharField('Title', max_length=1000)
    seendate = models.CharField('seendate', max_length=1000)
    socialimage = models.CharField('image', max_length=1000)
    domain = models.CharField('Domain', max_length=1000)
    language = models.CharField('Language', max_length=1000)
    sourcecountry = models.CharField('Countrysource', max_length=1000)
    finbert_positive = models.FloatField(default=None)
    finbert_negative = models.FloatField(default=None)
    final_finbert = models.FloatField(default=None)
    fama_french = models.IntegerField(default=0)

class Crypto(models.Model):
    crypto_name = models.CharField('Crypto', max_length=20, primary_key=True, unique=True)
    ticker = models.CharField('ticker',  max_length=20, unique=True)
    abnormal_rets = models.IntegerField('Farma French', default=0)
