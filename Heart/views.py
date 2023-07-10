from django.shortcuts import render

def index(request):
    return render(request,'index.html',{})


def logout(request):
    return render(request,'index.html',{})