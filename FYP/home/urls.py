from django.urls import path
from .views import homepage, login_view, dashboard, view_profile, signup, user_logout,view_homepage

urlpatterns = [
    path('', homepage, name='homepage'),          
    path('login/', login_view, name='login'),      
    path('dashboard/', dashboard, name='dashboard'),
    path('profile/', view_profile, name='view_profile'),
    path('signup/', signup, name='signup'),
    path('logout/', user_logout, name='logout'),
    path('viewhomepage/', view_homepage, name='view_homepage'),
]
