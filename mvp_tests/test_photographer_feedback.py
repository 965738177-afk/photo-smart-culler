from pathlib import Path
from zipfile import ZipFile
from xml.etree.ElementTree import Element, SubElement, tostring
import pytest

from scripts.evaluate_photographer_feedback import NS, read_form, feedback_metrics


def test_equivalence_credit_is_direct_not_entire_group():
    rows=[{'path':'human','human_retained':True}, {'path':'ai','human_retained':False},
          {'path':'other','human_retained':False}]
    notes=[{'interpretation':'equivalent_pair','ai_path':'ai','human_path':'human'}]
    result=feedback_metrics(rows,{'other'},notes)
    assert result['direct_equivalent_pair_credits']==0
    result=feedback_metrics(rows,{'ai','human'},notes)
    assert result['exact_or_direct_equivalent']==1
    assert not result['independent_75_acceptance']


def test_uncertain_duplicate_is_not_a_confirmed_positive():
    rows=[{'path':'a','human_retained':False}, {'path':'b','human_retained':True}]
    notes=[{'interpretation':'ambiguous_duplicate','ai_path':'a','human_path':None}]
    result=feedback_metrics(rows,{'a'},notes)
    assert result['confirmed_acceptable_extra_files']==0
    assert result['exact_or_direct_equivalent']==0


def test_only_explicit_accepted_file_changes_partial_reference():
    rows=[{'path':'a','human_retained':False}, {'path':'b','human_retained':True}]
    notes=[{'interpretation':'acceptable','ai_path':'a','human_path':None}]
    result=feedback_metrics(rows,{'a'},notes)
    assert result['confirmed_acceptable_extra_files']==1
    assert result['partially_adjusted_positive_count']==2
    assert result['exact_after_partial_feedback']==1


def form(tmp_path, *, caption='b.ARW', note='基本等价', heading='01 上午 组 4'):
    root=Element(NS+'document');body=SubElement(root,NS+'body')
    def paragraph(parent,text):
        p=SubElement(parent,NS+'p');r=SubElement(p,NS+'r');SubElement(r,NS+'t').text=text
    paragraph(body,heading)
    table=SubElement(body,NS+'tbl');row=SubElement(table,NS+'tr')
    for name in ['a.ARW',caption]:
        paragraph(SubElement(row,NS+'tc'),name)
    paragraph(body,'我的取舍原因：'+note)
    path=tmp_path/'form.docx'
    with ZipFile(path,'w') as z:
        z.writestr('word/document.xml',tostring(root))
    return path


SAMPLE={'folder':'/source/上午','group':'4','ai_path':'/source/上午/a.ARW',
        'human_path':'/source/上午/b.ARW'}


def test_form_verifies_group_folder_and_filenames(tmp_path):
    records=read_form(form(tmp_path),[SAMPLE])
    assert records[0]['interpretation']=='equivalent_pair'
    with pytest.raises(ValueError,match='identities'):
        read_form(form(tmp_path,caption='wrong.ARW'),[SAMPLE])
    with pytest.raises(ValueError,match='confirmation'):
        read_form(form(tmp_path,note='不等价'),[SAMPLE])
    with pytest.raises(ValueError,match='Unfinished'):
        read_form(form(tmp_path,note='［请在这里填写］'),[SAMPLE])
