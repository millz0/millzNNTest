using SimplePerceptron.OOP;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace SimplePerceptron.OOP
{
    /// <summary>
    /// ニューロンが持つ機能
    /// </summary>
    public interface INeuron
    {
        /// <summary>
        /// 名称
        /// </summary>
        string Name { get; set; }

        /// <summary>
        /// 出力値
        /// </summary>
        double OutputValue { get; set; }

        /// <summary>
        /// 順方向の伝搬を受信 
        /// </summary>
        /// <param name="neuron">入力側のニューロン</param>
        /// <param name="signal">入力値</param>
        void ReceiptFromForward(INeuron neuron, double signal);

        /// <summary>
        /// 逆方向の伝番を受信
        /// </summary>
        /// <param name="neuron">出力側のニューロン</param>
        /// <param name="signal">入力値</param>
        void ReceiptFromBackward(INeuron neuron, double signal);

        /// <summary>
        /// ニューロンを設定する
        /// </summary>
        /// <param name="inputNeurons">入力側ニューロンの配列</param>
        /// <param name="outputNeurons">出力側ニューロンの配列</param>
        void SetNeuron(INeuron[] inputNeurons, INeuron[] outputNeurons);

        /// <summary>
        /// ニューロンを設定する
        /// </summary>
        /// <param name="inputNeuron">入力側ニューロン</param>
        /// <param name="outputNeurons">出力側ニューロンの配列</param>
        void SetNeuron(INeuron inputNeuron, INeuron[] outputNeurons);

        /// <summary>
        /// ニューロンを設定する
        /// 
        /// </summary>
        /// <param name="inputNeurons">入力側ニューロンの配列</param>
        /// <param name="outputNeuron">出力側ニューロン</param>
        void SetNeuron(INeuron[] inputNeurons, INeuron outputNeuron);

        /// <summary>
        /// ニューロンを設定する
        /// </summary>
        /// <param name="inputNeuron">入力側ニューロン</param>
        /// <param name="outputNeuron">出力側ニューロン</param>
        void SetNeuron(INeuron inputNeuron, INeuron outputNeuron);
    }
}


namespace SimplePerceptron.OOP
{
    /// <summary>
    /// ニューロンの機能を提供するベースクラス
    /// </summary>
    public abstract class NeuronBase : INeuron
    {
        protected List<INeuron> _inputs;
        protected List<INeuron> _outputs;

        /// <summary>
        /// 名称
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// 出力値
        /// </summary>
        public double OutputValue { get; set; }

        public double ActivationFunction_df(INeuron n) => n.OutputValue * (1 - n.OutputValue); // Sigmoid関数 の微分

        /// <summary>
        /// ニューロンを設定する
        /// </summary>
        /// <param name="inputNeurons">入力側ニューロンの配列</param>
        /// <param name="outputNeurons">出力側ニューロンの配列</param>
        public void SetNeuron(INeuron[] inputNeurons, INeuron[] outputNeurons)
        {
            _inputs = inputNeurons.ToList();
            _outputs = outputNeurons.ToList();
            Initialize();
        }

        /// <summary>
        /// ニューロンを設定する
        /// </summary>
        /// <param name="inputNeuron">入力側ニューロン</param>
        /// <param name="outputNeurons">出力側ニューロンの配列</param>
        public void SetNeuron(INeuron inputNeuron, INeuron[] outputNeurons) => SetNeuron(new INeuron[] { inputNeuron }, outputNeurons);

        /// <summary>
        /// ニューロンを設定する
        /// 
        /// </summary>
        /// <param name="inputNeurons">入力側ニューロンの配列</param>
        /// <param name="outputNeuron">出力側ニューロン</param>
        public void SetNeuron(INeuron[] inputNeurons, INeuron outputNeuron) => SetNeuron(inputNeurons, new INeuron[] { outputNeuron });

        /// <summary>
        /// ニューロンを設定する
        /// </summary>
        /// <param name="inputNeuron">入力側ニューロン</param>
        /// <param name="outputNeuron">出力側ニューロン</param>
        public void SetNeuron(INeuron inputNeuron, INeuron outputNeuron) => SetNeuron(new INeuron[] { inputNeuron }, new INeuron[] { outputNeuron });

        /// <summary>
        /// 初期化します
        /// </summary>
        virtual protected void Initialize()
        {
        }

        /// <summary>
        /// 順方向の伝搬を受信 
        /// </summary>
        /// <param name="neuron">入力側のニューロン</param>
        /// <param name="signal">入力値</param>
        virtual public void ReceiptFromForward(INeuron neuron, double signal)
        { }

        /// <summary>
        /// 逆方向の伝番を受信
        /// </summary>
        /// <param name="neuron">出力側のニューロン</param>
        /// <param name="signal">入力値</param>
        virtual public void ReceiptFromBackward(INeuron neuron, double signal)
        { }

        /// <summary>
        /// 指定された数のインスタンスを取得する
        /// </summary>
        /// <typeparam name="T">作成するNeuronBaseの派生型</typeparam>
        /// <param name="number">インスタンス数</param>
        /// <param name="name">インスタンス化するニューロンの役割/名前</param>
        /// <returns></returns>
        public static T[] GetInstance<T>(int number, string name) where T : NeuronBase, new()
        {
            var _array = new T[number];
            for (int i = 0; i < number; i++)
                _array[i] = new T { Name = name + " : " + i.ToString() };
            return _array;
        }
    }
}


namespace SimplePerceptron.OOP
{
    /// <summary>
    /// 入力層のニューロン
    /// </summary>
    public class InputScalar : NeuronBase
    {
        /// <summary>
        /// 予測する(順方向伝搬)
        /// </summary>
        /// <param name="inputValue">入力値</param>
        public void Infer(double inputValue)
        {
            OutputValue = inputValue;
            _outputs.ForEach(x => x.ReceiptFromForward(this, inputValue));
        }
    }
}
namespace SimplePerceptron.OOP
{
    /// <summary>
    /// パーセプトロン
    /// </summary>
    public class Perceptron : NeuronBase
    {
        /// <summary>
        /// 乱数発生用 使いまわして一様にする
        /// </summary>
        private static Random rnd = new Random();
        private static double RandomNumber => 0;// (Math.Sign(rnd.NextDouble() - 0.5d) * rnd.NextDouble());

        private double _alpha = 0.9d;
        private double _eta = 0.01d;

        private List<double> _weight;
        private List<double> _weightModify;
        private List<double> TotalWeightModify;

        private Dictionary<INeuron, double> _forwardSignal = new Dictionary<INeuron, double>();
        private Dictionary<INeuron, double> _backwardSignal = new Dictionary<INeuron, double>();
        private double _bias = 0d;
        private double _biasModify = 0d;
        private double TotalBiasModify = 0d;

        override protected void Initialize()
        {
            _weight = new List<double>();
            _weightModify = new List<double>();
            TotalWeightModify = new List<double>();

            foreach (var input in _inputs)
            {
                _weight.Add(RandomNumber);
                _weightModify.Add(0);
                TotalWeightModify.Add(0);
            }
        }

        public void updateTotalWeight(int batchSize)
        {
            for (int i = 0; i < _inputs.Count; i++)
            {
                _weight[i] += (TotalWeightModify[i] / batchSize);
            }
            _bias += (TotalBiasModify / batchSize);

            TotalBiasModify = 0;
            TotalWeightModify.Clear();
            for (int i = 0; i < _inputs.Count; i++)
            {
                TotalWeightModify.Add(0);
            }
        }

        private double ActivationFunction(double value) => 1.0 / (1.0 + Math.Exp(-value)); // Sigmoid関数 

        /// <summary>
        /// 順方向の伝搬を受信 
        /// </summary>
        /// <param name="neuron">入力側のニューロン</param>
        /// <param name="signal">入力値</param>
        override public void ReceiptFromForward(INeuron neuron, double signal)
        {
            _forwardSignal[neuron] = signal;
            if (_forwardSignal.Count == _inputs.Count) // 入力側の刺激が全て届いたら伝搬する
                PropagateToForward();
        }

        /// <summary>
        /// 逆方向の伝番を受信
        /// </summary>
        /// <param name="neuron">出力側のニューロン</param>
        /// <param name="signal">入力値</param>
        override public void ReceiptFromBackward(INeuron neuron, double signal)
        {
            _backwardSignal[neuron] = signal;
            if (_backwardSignal.Count == _outputs.Count) // 出力側の刺激が全て届いたら伝搬する
                PropagateToBackward();
        }

        /// <summary>
        /// 順方向のパーセプトロンへの伝送
        /// </summary>
        /// <param name="signal"></param>
        private void PropagateToForward()
        {
            double _sum = 0.0d;
            for (int i = 0; i < _inputs.Count; i++)
            {
                _sum += _weight[i] * _forwardSignal[_inputs[i]];
            }
            _sum += _bias; // バイアスを足してn+1の行列にする
            OutputValue = ActivationFunction(_sum);
            foreach (var output in _outputs)
            {
                output.ReceiptFromForward(this, OutputValue);
            }
            _forwardSignal.Clear();
        }

        /// <summary>
        /// 逆方向のパーセプトロンへの伝搬
        /// </summary>
        private void PropagateToBackward()
        {
            for (int i = 0; i < _inputs.Count; i++)
            {
                double _sum = 0.0d;
                for (int j = 0; j < _outputs.Count; j++)
                {
                    _weightModify[i] = _eta * _backwardSignal[_outputs[j]] * _inputs[i].OutputValue + _alpha * _weightModify[i];
                    TotalWeightModify[i] += _weightModify[i];
                    var tmpWeight = _weight[i] + _weightModify[i];
                    _sum += _backwardSignal[_outputs[j]] * tmpWeight;
                    _biasModify = _eta * _backwardSignal[_outputs[j]] + _alpha * _biasModify;
                    TotalBiasModify += _biasModify;
                }
                _inputs[i].ReceiptFromBackward(this, (ActivationFunction_df(_inputs[i]) * _sum));
            }
            _backwardSignal.Clear();
        }
    }
}

namespace SimplePerceptron.OOP
{
    /// <summary>
    /// 出力層の値を取り出すためのニューロン
    /// </summary>
    public class OutputScalar : NeuronBase
    {
        /// <summary>
        /// 学習する(逆方向伝搬)
        /// </summary>
        /// <param name="teacherValue">教師データ</param>
        public double Train(double teacherValue)
        {
            var loss = (teacherValue - OutputValue);
            _inputs[0].ReceiptFromBackward(this, loss * ActivationFunction_df(this));
            return loss;
        }

        // protected double ErrorFunction(double teacherValue) => (teacherValue - OutputValue) * ActivationFunction_df(this); // 誤差関数

        /// <summary>
        /// 順方向の伝搬を受信 
        /// </summary>
        /// <param name="neuron">入力側のニューロン</param>
        /// <param name="signal">入力値</param>
        override public void ReceiptFromForward(INeuron neuron, double signal) => OutputValue = signal;
    }
}



namespace ConsoleApp1
{

    class InputData
    {
        public InputData(double pl, double pl2, double em, double em2, double tea = -1)
        {
            Pl = pl;
            Pl2 = pl2;
            Em = em;
            Em2 = em2;
            T = tea;

            InputValue.Add(pl);
            InputValue.Add(pl2);
            InputValue.Add(em);
            InputValue.Add(em2);
        }
        double Pl;
        double Pl2;
        double Em;
        double Em2;
        public double T;
        public List<double> InputValue = new List<double>();
    }



    internal class Program
    {
        static void Main(string[] args)
        {
            var result = new List<(double, double, double)>();
            for (int k = 0; k < 10; k++)
            {
                OOP(ref result);
            }

            foreach (var val in result)
            {
                Console.WriteLine($"結果={val.Item1.ToString("0.000")}  損失={val.Item2.ToString("0.000")}   最後の損失={val.Item3.ToString("0.000")}");
            }
            Console.ReadKey();
        }

        static void OOP(ref List<(double, double, double)> result)
        {
            // 以下の構成とする
            // 入力層 = 2, 隠れ層 = 2, 出力層=1 のNN
            var _inputNN = NeuronBase.GetInstance<InputScalar>(4, "入力層");
            var _hideNN = NeuronBase.GetInstance<Perceptron>(8, "隠れ層");
            var _outNN = NeuronBase.GetInstance<Perceptron>(1, "出力層");
            var _outputScalar = NeuronBase.GetInstance<OutputScalar>(1, "出力値");

            // ネットワークの構成
            foreach (var inputNN in _inputNN)
            {
                inputNN.SetNeuron((INeuron)null, _hideNN);
            }
            //_inputNN[0].SetNeuron((INeuron)null, _hideNN); // 入力:null  出力:h0,h1
            //_inputNN[1].SetNeuron((INeuron)null, _hideNN); // 入力:null  出力:h0,h1
            foreach (var hideNN in _hideNN)
            {
                hideNN.SetNeuron(_inputNN, _outNN);
            }
            //_hideNN[0].SetNeuron(_inputNN, _outNN); // 入力:i0,i1  出力:o0
            //_hideNN[1].SetNeuron(_inputNN, _outNN); // 入力:i0,i1  出力:o0

            _outNN[0].SetNeuron(_hideNN, _outputScalar[0]); // 入力:h1,h2  出力:s0
            _outputScalar[0].SetNeuron(_outNN[0], (INeuron)null); // 入力:o0  出力:null

            // 訓練用XORデータ入力＋出力（教師）
            // 入力1(X),入力2(Y),教師データ
            var trainings = new InputData[] {
                new InputData(100,100,100,100,1),
                new InputData(100,100, 10,100,1),
                new InputData(100, 10,100,100,0),
                new InputData( 10,100,100,100,0),
                new InputData( 10, 10, 10,10,1),
                new InputData( 10, 10,100, 50,0),
                new InputData( 10,100, 10,100,0),
                new InputData( 90, 90, 50,100,1),
                new InputData( 30, 10,100,100,0),
                new InputData(100,100, 20, 90,1),
                new InputData( 70,100, 10,100,1),
                new InputData( 10,100, 50,100,0),
                new InputData( 90, 90, 30,100,1),
                new InputData( 50, 50,100,100,0),
                new InputData( 50, 50, 50,100,0),
                new InputData( 40, 40, 50,100,0),
                new InputData( 80, 70, 70,100,1)
            }.ToList();

            // テスト用XOR入力
            // 入力1(X),入力2(Y)
            var tests = new InputData(90, 9, 70, 100, 0);

            //for (int j = 0; j < 10000; j++)
            //{
            //    // 学習+誤差逆伝送
            //    trainings.ForEach(x =>
            //    {
            //        for (int i = 0; i < _inputNN.Count(); i++)
            //            _inputNN[i].Infer(x.InputValue[i]);
            //        for (int i = 0; i < _outputScalar.Count(); i++)
            //            _outputScalar[i].Train(x.T);
            //    });
            //}

            // 学習のパラメータ
            int epochs = 1000; // エポック数
            int batchSize = 5; // ミニバッチのサイズ

            var rnd = new Random();
            double lastLoss = 100;
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // データセットをシャッフルする（過学習を防ぐため）
                trainings = trainings.OrderBy(x => rnd.Next()).ToList();

                var E = new List<double>();
                // ミニバッチごとに学習を行う
                for (int i = 0; i < trainings.Count; i += batchSize)
                {
                    // ミニバッチを作成
                    var miniBatch = trainings.Skip(i).Take(batchSize).ToList();

                    // ミニバッチごとに学習を行う
                    foreach (var data in miniBatch)
                    {
                        for (int j = 0; j < _inputNN.Count(); j++)
                            _inputNN[j].Infer(data.InputValue[j]);
                        for (int j = 0; j < _outputScalar.Count(); j++)
                        {
                            var val = _outputScalar[j].Train(data.T);
                            E.Add(val);
                        }
                    }

                    lastLoss = 0;
                    foreach (var e in E)
                    {
                        lastLoss += Math.Abs(e);
                    }
                    lastLoss /= E.Count;
                    Console.WriteLine($"{lastLoss.ToString("0.0000")}");

                    foreach (var data in _outNN)
                    {
                        data.updateTotalWeight(batchSize);
                    }
                    foreach (var data in _hideNN)
                    {
                        data.updateTotalWeight(batchSize);
                    }
                }

            }


            // 学習したNNを使用して予測(判定)の表示
            for (int i = 0; i < _inputNN.Count(); i++)
                _inputNN[i].Infer(tests.InputValue[i]);

            result.Add((_outputScalar[0].OutputValue, Math.Abs(_outputScalar[0].OutputValue - tests.T), lastLoss));

            Console.WriteLine($"結果=================={_outputScalar[0].OutputValue.ToString("0.00")}");
        }
    }
}

