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
@CreoleResource(name = "CoreTokenizer", comment = "Run CoreNLP pipeline tokenizer")
public class CoreTokenizer extends AbstractLanguageAnalyser implements ProcessingResource {
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
            System.out.println("using spanish");
            coreNlpProps.setProperty("tokenize.language", "es");
        }
        coreNlpProps.setProperty("annotators", "tokenize");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(coreNlpProps);
        CoreDocument document = new CoreDocument(this.document.getContent().toString());
        pipeline.annotate(document);
        for (CoreLabel token : document.tokens()) {
            int tokenStartIndex = token.beginPosition();
            int tokenEndIndex = token.endPosition();
            FeatureMap tokenFeatures = Factory.newFeatureMap();
            tokenFeatures.put("string",token.word());
            try {
                outputAS.add((long) tokenStartIndex, (long) tokenEndIndex,
                        "Token", tokenFeatures);
            } catch (InvalidOffsetException e) {
                e.printStackTrace();
            }
        }
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
