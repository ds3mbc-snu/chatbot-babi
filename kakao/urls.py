from django.urls import re_path
from .views import keyboard, message, numfact, block, demo, predict

app_name = 'kakao'

urlpatterns = [
    re_path(r'keyboard/', keyboard),
    re_path(r'^message$', message),
    re_path(r'^numfact$', numfact),
    re_path(r'^block$', block),
    re_path(r'^babi$', demo),
    re_path(r'^predict$', predict),

  	#path('get/story/' , get_story ),
    #path('get/answer/<int:question_idx>/<str:user_question>/', get_answer ),
]