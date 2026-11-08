"""Offline checks for the main dataset, output defaults, and safe collection."""
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import analyze_tfidf_biases as tf
import collect_gemini_responses as collector
from project_paths import PRIMARY_DATA, DEFAULT_PROMPTS


class MainWorkflowTests(unittest.TestCase):
    def test_saved_primary_dataset(self):
        records=tf.read_records(PRIMARY_DATA)
        self.assertEqual(len(records),75)
        self.assertEqual({p for _,_,p,_ in records},set(DEFAULT_PROMPTS))
        counts=Counter((run,tf.extract_subject(prompt,number)) for run,number,prompt,_ in records)
        self.assertEqual(len(counts),75)
        self.assertEqual(set(counts.values()),{1})
        self.assertEqual({run for run,_,_,_ in records},set(range(1,26)))

    def test_primary_tfidf_groups_and_quantifiers(self):
        records=tf.read_records(PRIMARY_DATA)
        rows=tf.analyze(records,None,True,False,10)
        self.assertEqual(len(rows),90)
        self.assertEqual(set(Counter((r['subject'],r['ngram_type']) for r in rows).values()),{10})
        tokens,_=tf.normalized_tokens('She wants more attention and a story with some drama.',True,False)
        self.assertIn('more',tokens)
        self.assertIn('some',tokens)
        self.assertTrue({'she','and','a','with'}.isdisjoint(tokens))

    def test_new_collection_uses_three_prompts(self):
        with TemporaryDirectory() as directory:
            output=Path(directory)/'new.txt'
            client=SimpleNamespace(interactions=SimpleNamespace(create=lambda **kwargs:SimpleNamespace(output_text='Example answer.')))
            with patch.object(collector.genai,'Client',return_value=client), \
                 patch('sys.argv',['collect','--runs','1','--output',str(output)]), \
                 patch.object(collector.time,'sleep') as sleep, redirect_stdout(StringIO()):
                collector.main()
            rows=tf.read_records(output)
            self.assertEqual(len(rows),3)
            self.assertEqual([r[2] for r in rows],list(DEFAULT_PROMPTS))
            sleep.assert_not_called()

    def test_existing_collection_is_preserved(self):
        with TemporaryDirectory() as directory:
            output=Path(directory)/'existing.txt'
            output.touch()
            before=output.read_bytes()
            with patch.object(collector.genai,'Client') as client, \
                 patch('sys.argv',['collect','--output',str(output)]):
                with self.assertRaises(FileExistsError):collector.main()
            self.assertEqual(output.read_bytes(),before)
            client.assert_not_called()


if __name__=='__main__':unittest.main()
