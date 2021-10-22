# Don't forget to support cases when target_text == ''
import editdistance


def calc_wer(target_text: str, pred_text: str):
    # if target_text == '':
    #     return len(pred_text.split())
    distance = editdistance.distance(target_text.split(), pred_text.split())
    return distance / (len(target_text.split()) + 1)


def calc_cer(target_text: str, pred_text: str):
    # if target_text == '':
    #     return len(pred_text)
    distance = editdistance.distance(target_text, pred_text)
    return distance / (len(target_text) + 1)
