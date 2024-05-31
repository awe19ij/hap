from django.urls import path
from .views import MyInterviewDetailView

urlpatterns = [
    path('<int:user_id>/<int:interview_id>/scripts/', MyInterviewDetailView.as_view(), name='my_interview_detail')
]

