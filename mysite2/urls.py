from django.urls import path, include

urlpatterns = [
    path('chatbot/', include('chatbot.urls', namespace='chatbot')),
]