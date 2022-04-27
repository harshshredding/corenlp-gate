package ca.concordia.gate;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.Tree;
import gate.*;
import gate.creole.AbstractLanguageAnalyser;
import gate.creole.ExecutionException;
import gate.creole.ResourceInstantiationException;
import gate.creole.metadata.CreoleParameter;
import gate.creole.metadata.CreoleResource;
import gate.creole.metadata.Optional;
import gate.creole.metadata.RunTime;
import gate.util.GateRuntimeException;
import gate.util.InvalidOffsetException;

import java.util.*;
import java.util.stream.Collectors;

/**
 * This plugin runs the CoreNLP pipeline on the document, generating "Token" and "Sentence" annotations.
 */
@CreoleResource(name = "CoreSentenceSplitter", comment = "Run CoreNLP pipeline tokenizer")
public class CoreSentenceSplitter extends AbstractLanguageAnalyser implements ProcessingResource {
    private String language;

    @Override
    public void reInit() throws ResourceInstantiationException {
        init();
    }

    /**
     * Initialise this resource, and return it.
     */
    @Override
    public Resource init() throws ResourceInstantiationException {
        return super.init();
    }

    public void execute() throws ExecutionException {
        if (this.document == null) throw new GateRuntimeException("No document to process!");
        System.out.println("document name:" + this.document.getName());
        System.out.println("document size" + this.document.getContent().size());
        System.out.println("Language:" + this.language);
        AnnotationSet outputAS = this.document.getAnnotations();
        Properties coreNlpProps = new Properties();
        if (language.equals("spanish")) {
            coreNlpProps.setProperty("tokenize.language","es");
            coreNlpProps.setProperty("pos.model","edu/stanford/nlp/models/pos-tagger/spanish-ud.tagger");
            coreNlpProps.setProperty("ner.model","edu/stanford/nlp/models/ner/spanish.ancora.distsim.s512.crf.ser.gz");
            coreNlpProps.setProperty("ner.applyNumericClassifiers","true");
            coreNlpProps.setProperty("ner.useSUTime","true");
            coreNlpProps.setProperty("ner.language","es");
            coreNlpProps.setProperty("sutime.language","spanish");
            coreNlpProps.setProperty("parse.model","edu/stanford/nlp/models/srparser/spanishSR.beam.ser.gz");
            coreNlpProps.setProperty("depparse.model","edu/stanford/nlp/models/parser/nndep/UD_Spanish.gz");
            coreNlpProps.setProperty("depparse.language", "spanish");
        }

        edu.stanford.nlp.pipeline.Annotation document = new edu.stanford.nlp.pipeline.Annotation(this.document.getContent().toString());
        addTokens(document);
        // we don't tokenize
        coreNlpProps.setProperty("annotators", "ssplit");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(coreNlpProps, false);
        pipeline.annotate(document);
        CoreDocument coreDocument = new CoreDocument(document);
        for (CoreSentence sentence : coreDocument.sentences()) {
            // First, add the sentence
            int sentenceStartIndex = sentence.tokens().get(0).beginPosition();
            int sentenceEndIndex = sentence.tokens().get(sentence.tokens().size() - 1).endPosition();
            try {
                outputAS.add((long) sentenceStartIndex, (long) sentenceEndIndex,
                        "Sentence", Factory.newFeatureMap());
            } catch (InvalidOffsetException e) {
                e.printStackTrace();
            }
        }
    }

    private void addTokens(edu.stanford.nlp.pipeline.Annotation document) {
        List<CoreLabel> tokenLabelList = new ArrayList<>();
        for (Annotation ano : this.document.getAnnotations()) {
            // filter for tokens
            if (ano.getType().equals("Token")) {
                String tokenString = (String)ano.getFeatures().get("string");
                CoreLabel tokenLabel = CoreLabel.wordFromString(tokenString);
                tokenLabel.set(CoreAnnotations.CharacterOffsetBeginAnnotation.class,
                        Math.toIntExact(ano.getStartNode().getOffset()));
                tokenLabel.set(CoreAnnotations.CharacterOffsetEndAnnotation.class,
                        Math.toIntExact(ano.getEndNode().getOffset()));
                tokenLabel.set(CoreAnnotations.TextAnnotation.class, tokenString);
                tokenLabel.set(CoreAnnotations.IsNewlineAnnotation.class, false);
                tokenLabel.set(CoreAnnotations.ValueAnnotation.class, tokenString);
                tokenLabelList.add(tokenLabel);
                System.out.println("Generated Token: " + tokenString + " (" +
                        ano.getStartNode().getOffset()+ "," + ano.getEndNode().getOffset() + ")");
            }
        }
        tokenLabelList = tokenLabelList.stream().sorted(Comparator
                        .comparingInt(o -> o.get(CoreAnnotations.CharacterOffsetBeginAnnotation.class)))
                .collect(Collectors.toList());
        document.set(CoreAnnotations.TokensAnnotation.class, tokenLabelList);
    }

    @RunTime
    @Optional
    @CreoleParameter(comment = "The language of the input text", defaultValue = "english")
    public void setLanguage(String language) {
        this.language = language;
    }

    public String getLanguage() {
        return this.language;
    }
}
