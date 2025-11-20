from django.contrib import admin
from django.urls import path
from core import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='home'),
    path('api/preview/', views.get_preview_data, name='preview'),
    path('api/train/', views.train_api, name='train'),
    path('api/sigmoid/', views.calculate_sigmoid, name='sigmoid'),
]