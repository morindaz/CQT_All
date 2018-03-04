#coding=utf-8
import os
import  xml.dom.minidom
import pickle
#from cut_function import cut_type

def cut_type(start,end,video,root_dir1,root_dir2, risk):
    start -= 1
    if start < 0:
        start = 0
    name = video + '-' + str(start) + '-' + str(end) + '.mov'
    newname = str(risk) + '-' + name
    duration = end - start
    os.system('ffmpeg -i /home/bresinno/pyAudioAnalysis/RMD/vedio/' + root_dir1 + '/' + video + '.mov -ss ' + str(start) + ' -t ' + str(duration) + ' /Users/ansir/Desktop/yh_video/old_videos/old_type_result/' + root_dir2 + '/' + newname)
    return newname

file_type = ['identity_ID_Card_number','identity_phone_number','identity_other','job_superficial'
                ,'job_deep','willingness','use','others_another_debit','others_other']
# question_type = ['ask - identity - ID Card number','ask - identity - phone number','ask - identity - other ','ask - job'
#                 ,'ask - willingness','ask - use','ask - others - another debit ','ask - others - other','ask - renhang',
#                 'answer - identity - ID Card number','answer - identity - phone number','answer - identity - other ',
#                 'answer - job','answer - willingness','answer - use','answer - others - another debit ','answer - others - other',
#                 'answer - renhang']
question_type = ['identity - ID Card number','identity - phone number','identity - other','job - superficial','job - deep','willingness','use','others - another debit','others - other']
def find_place(l1,l2,condition):
    return l2[[i for i, x in enumerate(l1) if x == condition][0]]


dirs = ['needtag170519','needtag170522']

for root_dir in dirs:
    # for tp in file_type:
    #     os.system('mkdir /Users/ansir/Desktop/yh_video/data/'+root_dir+'/'+tp)
    #     #os.system('mkdir /Users/ansir/Desktop/yh_video/data/'+root_dir+'/annotation')
    for _, _, files in os.walk('/home/bresinno/pyAudioAnalysis/RMD/vedio/'+root_dir+'/labels'):
        for filename in files:
            if filename.split('.')[-1] != 'anvil':
                continue

            dom = xml.dom.minidom.parse('/home/bresinno/pyAudioAnalysis/RMD/vedio/'+root_dir+'/labels/'+filename)
            tracks = dom.getElementsByTagName('track')

            # questions
            questions_dom = tracks[0].getElementsByTagName('el')
            questions = []
            for q in questions_dom:
                questions.append({
                    'date': root_dir,
                    'filename': filename.split('.')[:-1][0],
                    'start': q.getAttribute('start'),
                    'end': q.getAttribute('end'),
                    'type': q.getElementsByTagName('attribute')[0].firstChild.data,
                    'comment': q.getElementsByTagName('comment')[0].firstChild.data if len(q.getElementsByTagName('comment')) > 0 else '',
                    'risk': {'type':'no risk', 'comment':''}
                })

            # risks
            if len(tracks) > 1:         
                risks_dom = tracks[1].getElementsByTagName('el')
                for r in risks_dom:
                    index = int(r.getAttribute('index'))
                    questions[index]['risk']['type'] = r.getElementsByTagName('attribute')[0].firstChild.data
                    questions[index]['comment'] = r.getElementsByTagName('comment')[0].firstChild.data if len(r.getElementsByTagName('comment')) > 0 else ''


            for q in questions:
                risk_type = 0
                if q['risk']['type'] == 'suspectable risk':
                    risk_type = 1
                elif q['risk']['type'] == 'validated risk':
                    risk_type = 2
                print(risk_type)
               
                q['video'] = cut_type(float(q['start']), float(q['end']), q['filename'],root_dir,  find_place(question_type,file_type,q['type']), risk_type)
                
#             with open('/Users/ansir/Desktop/yh_video/data/'+root_dir+'/annotation/'+filename+'.pkl', 'wb') as f:
#                 pickle.dump(questions, f)