from django.contrib import admin
from django.urls import path
from App import views
urlpatterns = [
     path('',views.Home),
     path('Home/',views.Home),     
     path('Contact/',views.Contact),
     path('Donate/',views.Donations),
     path('Main/',views.Mainpage)
]


    