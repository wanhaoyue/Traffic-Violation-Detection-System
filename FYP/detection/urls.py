from django.urls import path
from . import views  

urlpatterns = [
    path('detect/', views.detect_image, name='detect_image'),
    path('detect-license-plate/', views.detect_license_plate, name='detect_license_plate'),
    path('violations/', views.view_violations, name='view_violations'),
    path('violations/delete/<int:record_id>/', views.delete_violation, name='delete_violation'),
    path('violations/clear/', views.clear_violations, name='clear_violations'),
    path('statistics/', views.statistics, name='statistics'),  
    path('statistics_data/', views.statistics_data, name='statistics_data'),
    path('get_recommendations/', views.get_recommendations, name='get_recommendations'),
]
