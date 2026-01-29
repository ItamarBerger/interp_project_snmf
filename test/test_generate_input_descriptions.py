import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from experiments.snmf_interp.generate_input_descriptions import run, make_generate_concept

@pytest.mark.asyncio
async def test_run_with_mocks():
    # Mock arguments
    args = MagicMock()
    args.input_json = "test_input.json"
    args.output_json = "test_output.json"
    args.layers = "0,6,12"
    args.k_values = "100"
    args.top_m = 5
    args.max_tokens = 100
    args.concurrency = 10
    args.retries = 3
    args.env_var = "TEST_API_KEY"
    args.model = "mock-model"

    # Mock data
    mock_data = [
        {"K": 100, "layer": 0, "level": 0, "h_row": 1, "top_activations": [{"token": "test", "context": "example", "activation": 0.9}]}
    ]

    # Mock API key
    with patch("os.getenv", return_value="mock-api-key"), \
         patch("experiments.snmf_interp.generate_input_descriptions.load_data", return_value=mock_data), \
         patch("experiments.snmf_interp.generate_input_descriptions.save_data") as mock_save_data, \
         patch("google.generativeai.GenerativeModel") as MockModel:

        # Mock the model's generate_content method
        mock_model_instance = MockModel.return_value
        mock_model_instance.generate_content = MagicMock(return_value=MagicMock(text="Mocked concept"))

        # Run the function
        await run(args)

        # Assertions
        mock_save_data.assert_called_once()
        saved_data = mock_save_data.call_args[0][1]
        assert len(saved_data) == 1
        assert saved_data[0]["description"] == "Mocked concept"