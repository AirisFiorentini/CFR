from Dudo_m_v import MDudoTrainer
import pytest



@pytest.mark.parametrize("iterations, expected_result", [(500, -7/258)])
def test_encode_file_name(iterations, expected_result):
    epsilon = 0.005
    result = MDudoTrainer().train(iterations)[1]
    assert abs(result - expected_result) < epsilon