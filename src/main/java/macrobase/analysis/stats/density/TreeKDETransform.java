package macrobase.analysis.stats.density;

import macrobase.analysis.pipeline.operator.MBStream;
import macrobase.analysis.transform.FeatureTransform;
import macrobase.conf.MacroBaseConf;
import macrobase.datamodel.Datum;

import java.util.List;

public class TreeKDETransform implements FeatureTransform{
    protected final MBStream<Datum> output = new MBStream<>();
    protected final MacroBaseConf mbConf;
    protected final TreeKDEConf tConf;

    public TreeKDETransform(MacroBaseConf mbConf, TreeKDEConf tConf) {
        this.mbConf = mbConf;
        this.tConf = tConf;
    }

    @Override
    public void consume(List<Datum> records) throws Exception {

    }

    @Override
    public MBStream<Datum> getStream() throws Exception {
        return output;
    }

    @Override
    public void initialize() {}
    @Override
    public void shutdown() {}
}
