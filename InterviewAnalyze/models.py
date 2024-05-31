from django.db import models
from QuestionList.models import QuestionLists
from django.conf import settings

class InterviewAnalysis(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    question_list = models.ForeignKey(QuestionLists, on_delete=models.CASCADE)  # QuestionLists 모델 참조

    # response_1부터 response_10까지 각 응답과 관련 정보 저장 필드
    for i in range(1, 11):
        locals()[f'response_{i}'] = models.TextField(blank=True, null=True)
        locals()[f'redundancies_{i}'] = models.TextField(blank=True, null=True)
        locals()[f'inappropriateness_{i}'] = models.TextField(blank=True, null=True)
        locals()[f'corrections_{i}'] = models.TextField(blank=True, null=True)
        locals()[f'corrected_response_{i}'] = models.TextField(blank=True, null=True)

    # 인터뷰 전체에 대한 총평을 저장할 필드 추가
    overall_feedback = models.TextField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)  # 생성 시각 자동 저장

    def __str__(self):
        return f'Responses for {self.question_list.id}'
