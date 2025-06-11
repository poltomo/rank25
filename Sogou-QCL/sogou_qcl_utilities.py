import bz2
import xml.etree.ElementTree as ET
from typing import Generator, Dict, Any, Tuple, List, Optional

import jieba
import pandas as pd

def stream_csv(filename: str, batch_size: int = 1024, skiprows: int = 1, delimiter: str = ',', column_names=None):
    """
    Stream CSV file in chunks, skipping a specified number of initial rows.

    Parameters:
    - filename: Path to the CSV file.
    - batch_size: Number of rows per chunk.
    - skiprows: Number of rows to skip at the beginning (e.g., 1 if the first line is not column headers).
    - delimiter: Field delimiter.
    - column_names: Optional list of column names to use for the DataFrame.
    """
    for chunk in pd.read_csv(
        filename,
        chunksize=batch_size,
        engine="c",
        skiprows=skiprows,
        delimiter=delimiter,
        header=None if column_names is not None else 'infer',
        names=column_names
    ):
        yield chunk

# stream Sogou-QCL human relevance annotations
def stream_csv_human_relevance_pairs(filename: str, batch_size: int = 1024, skiprows: int = 0):
    """
    Stream a CSV/TSV file and convert prefixed string columns to integers on the fly using converters.
    """
    converters = {
        0: lambda x: int(x[1:]),  # strip 'q' from first column
        1: lambda x: int(x[1:]),  # strip 'd' from second column
        2: int                   # third column is already numeric
    }

    for chunk in pd.read_csv(
        filename,
        chunksize=batch_size,
        engine="c",
        skiprows=skiprows,
        delimiter="\t",
        header=None,
        converters=converters,
        names=["qid", "did", "label"]
    ):
        yield chunk

def stream_sogou_qcl(filename: str) -> Generator[Dict[str, Any], None, None]:
    """
    Streams query-document pairs from a bz2 compressed XML file (SogouQCL format).

    Args:
        filename (str): The path to the .bz2 XML file.

    Yields:
        dict: A dictionary representing a single query entry, including its documents.
              Example structure:
              {
                  'query': '笑傲大将军',
                  'query_frequency': 26,
                  'query_id': 'q340',
                  'docs': [
                      {
                          'url': 'http://tieba.baidu.com/f?kw=%E7%AC%91%E5%82%B2%E5%A4%A7%E5%B0%86%E5%86%9B',
                          'doc_id': 'd3378',
                          'title': '笑傲大将军吧_百度贴吧',
                          'content': '0 没人吗? 河南人偷王建 6-4',
                          'html': None, # Will be present if <html> tag exists
                          'doc_frequency': None, # Will be present if <doc_frequency> tag exists
                          'relevance': { # Will be present if <relevance> tag exists
                              'TCM': 0.37604473478,
                              'DBN': 0.216979172374,
                              # ... other relevance metrics
                          }
                      },
                      # ... other documents for this query
                  ]
              }
    """
    # Open the bz2 file for reading in binary mode
    with bz2.open(filename, 'rb') as bz2f:
        # iterparse provides an efficient way to parse large XML files
        # It yields (event, element) tuples as it encounters start/end tags
        context = ET.iterparse(bz2f, events=('end',))

        # Iterate over the parsed elements
        for event, elem in context:
            # We are interested in the end of a <q> element
            if event == 'end' and elem.tag == 'q':
                query_data = {}
                query_data['query'] = elem.find('query').text if elem.find('query') is not None else None
                query_data['query_frequency'] = int(elem.find('query_frequency').text) if elem.find('query_frequency') is not None else None
                query_data['query_id'] = elem.find('query_id').text if elem.find('query_id') is not None else None

                docs = []
                for doc_elem in elem.findall('doc'):
                    doc_data = {}
                    doc_data['url'] = doc_elem.find('url').text if doc_elem.find('url') is not None else None
                    doc_data['doc_id'] = doc_elem.find('doc_id').text if doc_elem.find('doc_id') is not None else None
                    doc_data['title'] = doc_elem.find('title').text if doc_elem.find('title') is not None else None
                    doc_data['content'] = doc_elem.find('content').text if doc_elem.find('content') is not None else None
                    
                    # Handle optional <html> tag
                    html_elem = doc_elem.find('html')
                    doc_data['html'] = html_elem.text if html_elem is not None else None

                    # Handle optional <doc_frequency> tag
                    doc_freq_elem = doc_elem.find('doc_frequency')
                    doc_data['doc_frequency'] = int(doc_freq_elem.text) if doc_freq_elem is not None else None

                    # Handle optional <relevance> tag and its children
                    relevance_elem = doc_elem.find('relevance')
                    if relevance_elem is not None:
                        relevance_scores = {}
                        for score_elem in relevance_elem:
                            relevance_scores[score_elem.tag] = float(score_elem.text)
                        doc_data['relevance'] = relevance_scores
                    else:
                        doc_data['relevance'] = None
                    
                    docs.append(doc_data)
                query_data['docs'] = docs
                
                yield query_data

                # Clear the element from memory once processed to free up resources
                elem.clear()

# better payload for graph structures
def stream_sogou_qcl2(filename: str) -> Generator[Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]], None, None]:
    """
    Streams query data, document data, and relevance edges from a bz2 compressed XML file (SogouQCL format).

    Args:
        filename (str): The path to the .bz2 XML file.

    Yields:
        tuple: A tuple containing three elements:
               - dict: A dictionary representing a single query entry.
               - list: A list of dictionaries, each representing a document associated with the query.
               - list: A list of dictionaries, each representing an edge (query-document relevance).

        Example structure for a single yielded tuple:
        (
            {
                'query': '笑傲大将军',
                'query_frequency': 26,
                'query_id': 340
            },
            [
                {
                    'url': 'http://tieba.baidu.com/f?kw=%E7%AC%91%E5%82%B2%E5%A4%A7%E5%B0%86%E5%86%9B',
                    'doc_id': 3378,
                    'title': '笑傲大将军吧_百度贴吧',
                    'content': '0 没人吗? 河南人偷王建 6-4',
                    'html': None,
                    'doc_frequency': None
                },
                # ... other documents for this query
            ],
            [
                {
                    'query_id': 340,
                    'doc_id': 3378,
                    'relevance': {
                        'TCM': 0.37604473478,
                        'DBN': 0.216979172374,
                        # ... other relevance metrics
                    }
                },
                # ... other edges for this query
            ]
        )
    """
    with bz2.open(filename, 'rb') as bz2f:
        context = ET.iterparse(bz2f, events=('end',))

        for event, elem in context:
            if event == 'end' and elem.tag == 'q':
                # Initialize lists for current query's data
                current_query_payload = {}
                current_docs_payloads = []
                current_edges = []

                # --- Process Query Data ---
                query_id_str = elem.find('query_id').text if elem.find('query_id') is not None else None
                current_query_payload['query'] = elem.find('query').text if elem.find('query') is not None else None
                current_query_payload['query_frequency'] = int(elem.find('query_frequency').text) if elem.find('query_frequency') is not None else None
                current_query_payload['query_id'] = int(query_id_str[1:]) if query_id_str and query_id_str.startswith('q') else None


                # --- Process Document and Edge Data ---
                for doc_elem in elem.findall('doc'):
                    doc_data = {}
                    edge_data = {}

                    doc_id_str = doc_elem.find('doc_id').text if doc_elem.find('doc_id') is not None else None
                    
                    # Document Payload
                    doc_data['url'] = doc_elem.find('url').text if doc_elem.find('url') is not None else None
                    doc_data['doc_id'] = int(doc_id_str[1:]) if doc_id_str and doc_id_str.startswith('d') else None
                    doc_data['title'] = doc_elem.find('title').text if doc_elem.find('title') is not None else None
                    doc_data['content'] = doc_elem.find('content').text if doc_elem.find('content') is not None else None
                    
                    html_elem = doc_elem.find('html')
                    doc_data['html'] = html_elem.text if html_elem is not None else None

                    doc_freq_elem = doc_elem.find('doc_frequency')
                    doc_data['doc_frequency'] = int(doc_freq_elem.text) if doc_freq_elem is not None else None

                    current_docs_payloads.append(doc_data)

                    # Edge Payload (Relevance Scores)
                    relevance_elem = doc_elem.find('relevance')
                    if relevance_elem is not None:
                        relevance_scores = {}
                        for score_elem in relevance_elem:
                            relevance_scores[score_elem.tag] = float(score_elem.text)
                        
                        edge_data['query_id'] = current_query_payload['query_id']
                        edge_data['doc_id'] = doc_data['doc_id']
                        edge_data['relevance'] = relevance_scores
                        current_edges.append(edge_data)
                    
                yield current_query_payload, current_docs_payloads, current_edges
                
                elem.clear()

def stream_sogou_qcl3(filename: str) -> Generator[Tuple[Tuple[int, Dict[str, Any]], List[Tuple[int, Dict[str, Any]]], List[Dict[str, Any]]], None, None]:
    """
    Streams query data, document data, and relevance edges from a bz2 compressed XML file (SogouQCL format).

    Args:
        filename (str): The path to the .bz2 XML file.

    Yields:
        tuple: A tuple containing three elements:
               - tuple: A tuple containing the numeric query ID and a dictionary representing a single query entry (without the ID).
               - list: A list of tuples, each containing the numeric document ID and a dictionary representing a document associated with the query (without the ID).
               - list: A list of dictionaries, each representing an edge (query-document relevance).

        Example structure for a single yielded tuple:
        (
            (
                340, # query_id
                {
                    'query': '笑傲大将军',
                    'query_frequency': 26
                }
            ),
            [
                (
                    3378, # doc_id
                    {
                        'url': 'http://tieba.baidu.com/f?kw=%E7%AC%91%E5%82%B2%E5%A4%A7%E5%B0%86%E5%86%9B',
                        'title': '笑傲大将军吧_百度贴吧',
                        'content': '0 没人吗? 河南人偷王建 6-4',
                        'html': None,
                        'doc_frequency': None
                    }
                ),
                # ... other documents for this query
            ],
            [
                {
                    'query_id': 340,
                    'doc_id': 3378,
                    'relevance': {
                        'TCM': 0.37604473478,
                        'DBN': 0.216979172374,
                        # ... other relevance metrics
                    }
                },
                # ... other edges for this query
            ]
        )
    """
    with bz2.open(filename, 'rb') as bz2f:
        context = ET.iterparse(bz2f, events=('end',))

        for event, elem in context:
            if event == 'end' and elem.tag == 'q':
                # Initialize lists for current query's data
                current_query_id: int | None = None
                current_query_payload_dict = {}
                current_docs_payloads_with_ids = [] # To store (doc_id, doc_payload_dict)
                current_edges = []

                # --- Process Query Data ---
                query_id_str = elem.find('query_id').text if elem.find('query_id') is not None else None
                if query_id_str and query_id_str.startswith('q'):
                    current_query_id = int(query_id_str[1:])

                current_query_payload_dict['query'] = elem.find('query').text if elem.find('query') is not None else None
                current_query_payload_dict['query_frequency'] = int(elem.find('query_frequency').text) if elem.find('query_frequency') is not None else None

                # --- Process Document and Edge Data ---
                for doc_elem in elem.findall('doc'):
                    doc_data = {}
                    edge_data = {}
                    current_doc_id: int | None = None

                    doc_id_str = doc_elem.find('doc_id').text if doc_elem.find('doc_id') is not None else None
                    if doc_id_str and doc_id_str.startswith('d'):
                        current_doc_id = int(doc_id_str[1:])
                    
                    # Document Payload
                    doc_data['url'] = doc_elem.find('url').text if doc_elem.find('url') is not None else None
                    doc_data['title'] = doc_elem.find('title').text if doc_elem.find('title') is not None else None
                    doc_data['content'] = doc_elem.find('content').text if doc_elem.find('content') is not None else None
                    
                    html_elem = doc_elem.find('html')
                    doc_data['html'] = html_elem.text if html_elem is not None else None

                    doc_freq_elem = doc_elem.find('doc_frequency')
                    doc_data['doc_frequency'] = int(doc_freq_elem.text) if doc_freq_elem is not None else None

                    if current_doc_id is not None:
                        current_docs_payloads_with_ids.append((current_doc_id, doc_data))

                    # Edge Payload (Relevance Scores)
                    relevance_elem = doc_elem.find('relevance')
                    if relevance_elem is not None and current_query_id is not None and current_doc_id is not None:
                        relevance_scores = {}
                        for score_elem in relevance_elem:
                            relevance_scores[score_elem.tag] = float(score_elem.text)
                        
                        edge_data['query_id'] = current_query_id
                        edge_data['doc_id'] = current_doc_id
                        edge_data['relevance'] = relevance_scores
                        current_edges.append(edge_data)
                    
                # Yield the query ID and its payload as a tuple,
                # the list of document ID/payload tuples, and the list of edges
                if current_query_id is not None:
                    yield (current_query_id, current_query_payload_dict), current_docs_payloads_with_ids, current_edges
                
                elem.clear()