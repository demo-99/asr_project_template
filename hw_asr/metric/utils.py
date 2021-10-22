# Don't forget to support cases when target_text == ''
import editdistance


def calc_wer(target_text: str, pred_text: str):
    if target_text == '':
        target_text = ' '
    distance = editdistance.distance(target_text.split(), pred_text.split())
    return distance / len(target_text.split())


def calc_cer(target_text: str, pred_text: str):
    if target_text == '':
        target_text = ' '
    distance = editdistance.distance(target_text, pred_text)
    return distance / len(target_text)
