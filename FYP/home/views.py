from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout

# Hardcoded admin credentials
ADMIN_USERID = "a"
ADMIN_PASSWORD = "a"

def homepage(request):
    list(messages.get_messages(request))
    return render(request, "home/homepage.html")  

def view_homepage(request):
    return render(request, "home/viewhomepage.html") 

def login_view(request):
    messages.get_messages(request).used = True

    if request.method == 'POST':
        username = request.POST['user_id']
        password = request.POST['password']

        # User login check
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            request.session['user_id'] = username
            request.session['password'] = password  # Store plain-text password for profile view
            messages.success(request, f'{username} login successful!')
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid User ID or Password. Please try again.')
            return redirect('login')

    return render(request, "home/index.html")

def dashboard(request):
    if 'user_id' not in request.session:
        return redirect('login')
    username = request.session.get('user_id', 'Unknown')
    return render(request, "home/dashboard.html", {'username': username})

def view_profile(request):
    if 'user_id' not in request.session:
        return redirect('login')

    username = request.session.get('user_id', 'Unknown')
    password = request.session.get('password', 'Unknown')
    return render(request, "home/profile.html", {'username': username, 'password': password})

def signup(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        if User.objects.filter(username=username).exists():
            messages.error(request, 'User ID already exists!')
            return redirect('signup')

        user = User.objects.create_user(username=username, password=password)
        user.save()
        messages.success(request, 'User registered successfully!')
        return redirect('login')
    return render(request, 'home/signup.html')

def user_logout(request):
    if 'user_id' in request.session:
        username = request.session.get('user_id')
        # Clear session data
        logout(request)
        # Add success message
        messages.success(request, f'{username} logout successful!')
    return redirect('homepage')  # Redirect to homepage after logout
