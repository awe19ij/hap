from django.urls import path
from .views import start_gaze_tracking_view, stop_gaze_tracking_view

urlpatterns = [
    path('start/', start_gaze_tracking_view, name='start_gaze_tracking'),
    path('stop/', stop_gaze_tracking_view, name='stop-gaze-tracking'),
]
