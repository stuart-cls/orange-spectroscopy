from orangecontrib.spectroscopy.tests.test_owpreprocess import PreprocessorEditorTest
from orangecontrib.spectroscopy.widgets.owpreprocess import OWPreprocess
from orangecontrib.spectroscopy.tests.test_preprocess import SMALL_COLLAGEN
from orangecontrib.spectroscopy.widgets.preprocessors.spikeremoval import SpikeRemovalEditor
from orangecontrib.spectroscopy.preprocess import Despike



class TestSpikeRemovalEditor(PreprocessorEditorTest):

    def get_preprocessor(self):
        out = self.get_output(self.widget.Outputs.preprocessor)
        return out.preprocessors[0]
    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        self.editor = self.add_editor(SpikeRemovalEditor, self.widget)
        self.data = SMALL_COLLAGEN
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_for_preview()  # ensure initialization with preview data
    def test_no_interaction(self):
        self.widget.unconditional_commit()
        self.wait_until_finished()
        p = self.get_preprocessor()
        self.assertIsInstance(p, Despike)

    def test_basic(self):
        self.editor.dis = 5
        self.editor.cutoff = 100
        self.editor.threshold = 7
        self.widget.unconditional_commit()
        self.wait_until_finished()
        p = self.get_preprocessor()
        self.process_events()
        self.assertEqual(p.dis, 5)
        self.assertEqual(p.cutoff, 100)
        self.assertEqual(p.threshold, 7)
