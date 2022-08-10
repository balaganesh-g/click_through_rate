import pytest
from app import form_response

params_path = "params.yaml"


class NotANumber(Exception):
    def __init__(self, message="Values entered are not Numerical"):
        self.message = message
        super().__init__(self.message)


@pytest.fixture
def data():
    return {
        'correct_data': {'C1': 1005,
                         'banner_pos': 0,
                         'device_type': 1,
                         'device_conn_type': 0,
                         'C14': 16615,
                         'C15': 320,
                         'C16': 50,
                         'C17': 1863,
                         'C18': 3,
                         'C19': 39,
                         'C20': -1,
                         'C21': 23,
                         'site_id': '6256f5b4',
                         'site_domain': '28f93029',
                         'site_category': 'f028772b',
                         'app_id': 'ecad2386',
                         'app_domain': '7801e8d9',
                         'app_category': '07d7df22',
                         'device_id': 'a99f214a',
                         'device_ip': 'd5f8da08',
                         'device_model': '07d76b42'
                         },
        'incorrect_data': {'C1': 1005,
                           'banner_pos': 0,
                           'device_type': '1,',
                           'device_conn_type': 0,
                           'C14': 16615,
                           'C15': 320,
                           'C16': 50,
                           'C17': 1863,
                           'C18': 3,
                           'C19': 39,
                           'C20': -1,
                           'C21': 23,
                           'site_id': '6256f5b4',
                           'site_domain': '28f93029',
                           'site_category': 'f028772b',
                           'app_id': 'ecad2386',
                           'app_domain': '7801e8d9',
                           'app_category': '07d7df22',
                           'device_id': 'a99f214a',
                           'device_ip': 'd5f8da08',
                           'device_model': '07d76b42'
                           }
    }


def test_form_response_incorrect_values(data):
    res = form_response(data["incorrect_data"], params_path)
    assert res == NotANumber().message
