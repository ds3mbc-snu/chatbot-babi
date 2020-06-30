from django.urls import path, include

urlpatterns = [
    path('kakao/', include('kakao.urls', namespace='kakao')),
]
