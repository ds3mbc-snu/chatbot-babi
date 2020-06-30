from django.urls import re_path
from .views import keyboard, message

app_name = 'chatbot'

urlpatterns = [
    re_path(r'keyboard/', keyboard),
    re_path(r'^message$', message),
]