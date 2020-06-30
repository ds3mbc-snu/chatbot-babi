from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import requests
import json
import random

#import json

def keyboard(request):
    return JsonResponse({
        'type': 'text'
    })

@csrf_exempt
def block(request):
    answer = ((request.body).decode('utf-8'))
    return_json_str = json.loads(answer)
    return_str = return_json_str['userRequest']['block']['id']

    return JsonResponse({
        'version': "2.0",
        'template': {
            'outputs': [{
                'basicCard' : {
                    'title': "block Id",
                    'description': return_str
                }
            }]
        }
    })

@csrf_exempt
def message(request):
    #    answer = ((request.body).decode('utf-8'))
    #    return_json_str = json.loads(answer)
    #    return_str = return_json_str['userRequest']['utterance']

    return JsonResponse({
        'version': "2.0",
        'template': {
            'outputs': [{
                'simpleText': {
                    'text': "테스트 성공입니다."
                }
            }],
            'quickReplies': [{
                'label': '처음으로',
                'action': 'message',
                'messageText': '처음으로'
            }]
        }
    })

@csrf_exempt
def numfact(request):
    answer = ((request.body).decode('utf-8'))
    return_json_str = json.loads(answer)
    return_str = return_json_str['userRequest']['utterance']
    print(return_str)

    go_main_button = '처음으로'

    if return_str == '무작위 숫자':
        req = str(random.randrange(0,2001))
        url = "https://numbersapi.p.rapidapi.com/" + req + "/math"
    elif return_str == '직접 입력':
        req = return_json_str["action"]["params"]["positive_integer"]
        url = "https://numbersapi.p.rapidapi.com/" + req + "/math"

    querystring = {"fragment":"true","json":"true"}

    headers = {
        'x-rapidapi-host': "numbersapi.p.rapidapi.com",
        'x-rapidapi-key': "3af875ff19msh198ede237e09aa1p11485fjsn70319b00faf7"
        }

    response = requests.request("GET", url, headers=headers, params=querystring)

    return JsonResponse(
            {
                'version': '2.0',
                'template': {
                    'outputs': [
                        {
                            'simpleText': {
                                'text': req+': '+ response.text
                            }
                        }
                    ],
                    'quickReplies': [
                        {
                            'label': go_main_button,
                            'action': 'message',
                            'messageText': ""
                        },
                        {
                            'action': 'message',
                            'label': '한번 더?',
                            'messageText': '한번 더!',
                        }
                    ]
                }
            })


import pickle
import sys
sys.path.append('/home/ds3mbc/mysite/kakao/babi')

import numpy as np
from studybabi_kor import MemN2N

memn2n = None

@csrf_exempt
def demo(request):
    """
    Console-based demo
    """
    global memn2n, test_story, test_questions, test_qstory, question_idx, story_idx, last_sentence_idx, story_txt, question_txt, correct_answer

    data_dir = '/home/ds3mbc/mysite/kakao/babi/test_data_o.pkl'
    model_file = '/home/ds3mbc/mysite/kakao/babi/memn2n_study_kor.pklz'

    if memn2n == None:
        memn2n = MemN2N(data_dir, model_file)

        # Try to load model
        memn2n.load_model()

        # Read test data, which is saved by pickle
        with open(data_dir, 'rb') as f:
            test_story, test_questions, test_qstory = pickle.load(f)

    # Pick a random question
    question_idx = np.random.randint(test_questions.shape[1])
    story_idx = test_questions[0, question_idx]
    last_sentence_idx = test_questions[1, question_idx]

    # Get story and question
    story_txt, question_txt, correct_answer = memn2n.get_story_texts(
        test_story, test_questions, test_qstory,
        question_idx, story_idx, last_sentence_idx)

    stories = '\n'.join(story_txt)
    sug_question = question_txt

    return JsonResponse({
        'version': "2.0",
        'data': {
            'stories': stories,
            'sug_question': sug_question
        }
    })


@csrf_exempt
def predict(request):
    answer = ((request.body).decode('utf-8'))
    return_json_str = json.loads(answer)
    return_str = return_json_str['action']['params']['sys_text']

    if return_str == '추천 질문':
        pred_answer_idx, pred_prob, _ = memn2n.predict_answer(test_story, test_questions, test_qstory, question_idx, story_idx, last_sentence_idx, user_question='')
    else:
        pred_answer_idx, pred_prob, _ = memn2n.predict_answer(test_story, test_questions, test_qstory, question_idx, story_idx, last_sentence_idx, user_question=return_str)

    pred_answer = memn2n.reversed_dict[pred_answer_idx]

    res = "대답: '{pa}'\n정확도: {cs:0.2f}%".format(pa=pred_answer, cs=pred_prob*100)

    if return_str == '추천 질문' or return_str[:-1] == question_txt:
        if pred_answer == correct_answer:
            res += '\n정답!!'
        else:
            res += "\n틀렸습니다.. 정답: '{}'".format(correct_answer)

    return JsonResponse({
        'version': '2.0',
                'template': {
                    'outputs': [
                        {
                            'simpleText': {
                                'text': res
                            }
                        }
                    ],
                    'quickReplies': [
                        {
                            'action': 'block',
                            'label': '다른 질문',
                            'messageText': '다른 질문',
                            'blockId': '5efab7f31276c3000178b2b7'
                        },
                        {
                            'action': 'block',
                            'label': '다른 스토리',
                            'messageText': '다른 스토리',
                            'blockId': '5ee1adca2ca48c00011fd129'
                        },
                        {
                            'action': 'block',
                            'label': '처음으로',
                            'messageText': '처음으로',
                            'blockId': '5ee02b746fe05800015f38b4'
                        }
                    ]
                }
         })