from django.urls import path,include
from .views import *

urlpatterns = [
    
   path('', index, name='index'),
   path('camera-feed/', camera_feed, name='camera_feed'),
]