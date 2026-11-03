import pytest

from rmy.utils import label_rmy_header

HEADER = [
    'LOCATION,Boston,MA,USA,SRC-TMYx,725090,42.36,-71.01,-5.0,3.7\n',
    'DESIGN CONDITIONS,0\n',
    'TYPICAL/EXTREME PERIODS,0\n',
    'GROUND TEMPERATURES,0\n',
    'HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0\n',
    'COMMENTS 1,"Period of Record=2009-2023"\n',
    'COMMENTS 2,"Downloaded from Climate.Onebuilding.org"\n',
    'DATA PERIODS,1,1,Data,Sunday,1/ 1,12/31\n',
]


def test_only_comments_2_changes_and_field_counts_hold():
    new = label_rmy_header(HEADER, range(2009, 2024), 2022, 2015)
    assert new[:6] == HEADER[:6] and new[7] == HEADER[7]
    assert new[6] == ('COMMENTS 2,"RMY extreme weather file: heat waves and cold spells from '
                      'AMYs 2009-2023 inserted (peak heat year 2022; peak cold year 2015); '
                      'Downloaded from Climate.Onebuilding.org"\n')
    assert [l.count(',') for l in new] == [l.count(',') for l in HEADER]


def test_gaps_in_the_amy_years_are_listed():
    new = label_rmy_header(HEADER, [2010, 2011, 2013, 2014, 2016])
    assert 'AMYs 2010-2016 inserted (no AMY for 2012 2015)' in new[6]


@pytest.mark.parametrize('line, expected', [
    ('COMMENTS 2,\r\n', 'COMMENTS 2,RMY extreme weather file: heat waves and cold spells from AMYs 2020-2021 inserted\r\n'),
    ('COMMENTS 2,""\n', 'COMMENTS 2,"RMY extreme weather file: heat waves and cold spells from AMYs 2020-2021 inserted"\n'),
    ('COMMENTS 2,plain text\n', 'COMMENTS 2,RMY extreme weather file: heat waves and cold spells from AMYs 2020-2021 inserted; plain text\n'),
])
def test_empty_unquoted_and_crlf_comment_lines(line, expected):
    header = HEADER[:6] + [line] + HEADER[7:]
    assert label_rmy_header(header, [2020, 2021])[6] == expected


def test_missing_comments_2_raises():
    with pytest.raises(ValueError):
        label_rmy_header(HEADER[:6] + HEADER[7:], [2020])
