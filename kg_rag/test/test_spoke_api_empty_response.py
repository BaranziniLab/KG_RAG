"""
Unit test for issue #42: IndexError when SPOKE API returns empty list for a node.
https://github.com/BaranziniLab/KG_RAG/issues/42
"""
import unittest
from unittest.mock import patch, MagicMock


class TestGetContextUsingSpokeApiEmptyResponse(unittest.TestCase):

    @patch("kg_rag.utility.get_spoke_api_resp")
    def test_empty_node_context_does_not_raise(self, mock_api):
        """When SPOKE API returns an empty list, function should not raise IndexError."""
        from kg_rag.utility import get_context_using_spoke_api

        # First call returns types endpoint data
        types_response = MagicMock()
        types_response.json.return_value = {
            "nodes": {"Disease": {}, "Protein": {}, "Compound": {}},
            "edges": {"DaG": {}, "CtD": {}},
        }

        # Second call (neighborhood endpoint) returns empty list — the bug scenario
        empty_response = MagicMock()
        empty_response.json.return_value = []

        mock_api.side_effect = [types_response, empty_response]

        context, df = get_context_using_spoke_api("neurofibromatosis 2")

        self.assertEqual(context, "")
        self.assertTrue(df.empty)

    @patch("kg_rag.utility.get_spoke_api_resp")
    def test_non_empty_node_context_includes_identifier(self, mock_api):
        """When SPOKE API returns data, identifier info is appended to context."""
        from kg_rag.utility import get_context_using_spoke_api

        types_response = MagicMock()
        types_response.json.return_value = {
            "nodes": {"Disease": {}, "Gene": {}},
            "edges": {"DaG_association": {}},
        }

        # Edge types in SPOKE use underscores (e.g. "DaG_association")
        # Nodes have no underscore in neo4j_type
        node_response = MagicMock()
        node_response.json.return_value = [
            {
                "data": {
                    "neo4j_type": "Disease",
                    "id": "D001",
                    "properties": {
                        "name": "psoriasis",
                        "identifier": "DOID:8893",
                        "source": "DOID",
                    },
                }
            },
        ]

        mock_api.side_effect = [types_response, node_response]

        context, df = get_context_using_spoke_api("psoriasis")

        # Identifier info should be appended when node_context is non-empty
        self.assertIn("DOID", context)
        self.assertIn("DOID:8893", context)


if __name__ == "__main__":
    unittest.main()
