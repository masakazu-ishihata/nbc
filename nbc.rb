#!/usr/bin/env ruby

################################################################################
# default
################################################################################
# constants
@file = nil  # data file
@k = 3       # # clusters
@n = 5       # # features
@m = 100     # # samples
@i = 10      # # restart
@y = nil     # index of the answer feature
@I = []      # indice ignored

# flags
@show  = false
@scale = false

################################################################################
# Arguments
################################################################################
require "optparse"
OptionParser.new { |opts|
  # options
  opts.on("-h","--help","Show this message") {
    puts opts
    exit
  }
  opts.on("-f [string]", "data file"){ |f|
    @file = f
  }
  opts.on("-k [int]", "# clasters"){ |f|
    @k = f.to_i
  }
  opts.on("-n [int]", "# attributes"){ |f|
    @n = f.to_i
  }
  opts.on("-m [int]", "# samples"){ |f|
    @m = f.to_i
  }
  opts.on("-i [int]", "# random restart"){ |f|
    @i = f.to_i
  }
  opts.on("-y [int]", "index of class label in the data file"){ |f|
    @y = f.to_i
  }
  opts.on("-I [int,int,...]", "index should be ignored"){ |f|
    @I = f.split(",").map{|i| i.to_i}
    @I.sort!
    @I.reverse!
  }
  opts.on("--show", "show clustering results"){
    @show = true
  }
  opts.on("--scale", "scale data into [0, 1] range"){
    @scale = true
  }
  # parse
  opts.parse!(ARGV)
}

################################################################################
# Component
################################################################################
class MyComponent
  def learn(_d)
    # Viterbi learning
  end

  def sample(_n)
    # sample _n instances
  end

  def likelihood(_x)
    # likelihood
  end

  def log_likelihood(_x)
    # log likelihood
  end

  def type
    # type of this component
  end
end

########################################
# Categorical
########################################
class MyCategorical < MyComponent
  #### new ####
  def initialize(_vals)
    @n = _vals.size
    @p = Array.new(@n){|p| rand}
    sum = 0
    @p.each do |p|
      sum += p
    end
    @p.map!{|p| p/sum}

    @v2i = Hash.new
    @i2v = Array.new
    for i in 0.._vals.size-1
      v = _vals[i]
      @v2i[v] = i
      @i2v[i] = v
    end
  end

  #### sample ####
  def sample(_n)
    ss = []
    for i in 0.._n-1
      ss.push( sample_i )
    end
    ss
  end
  def sample_i
    s = 0
    r = rand
    for i in 0..@n-1
      s += @p[i]
      break if r < s
    end
    @i2v[i]
  end

  #### learn ####
  def learn(_d)
    c = Array.new(@n){|i| 0}
    _d.each do |v|
      c[ @v2i[v] ] += 1
    end
    for i in 0..@n-1
      @p[i] = c[i] / _d.size.to_f
    end
  end

  #### likelihood ####
  def likelihood(_x)
    @p[ @v2i[_x] ]
  end
  def log_likelihood(_x)
    Math::log( @p[ @v2i[_x] ] )
  end

  #### type ####
  def type
    "Categorical: #{@i2v} = #{@p}"
  end
end

########################################
# Gaussian
########################################
class MyGaussian < MyComponent
  #### new ####
  def initialize(_u, _s)
    @u = _u
    @s = _s
  end
  attr_reader :u, :s

  def set_ave(_u)
    @u = _u
  end
  def set_var(_s)
    @s = _s
  end

  #### learn parameters from _d ####
  def learn(_d)
    # average
    @u = 0
    _d.each do |d|
      @u += d
    end
    @u /= _d.size

    # variance
    @s = 0
    _d.each do |d|
      @s += (@u - d)**2
    end
    @s /= _d.size
  end

  #### sample ####
  def sample(_n)
    ss = []
    for i in 0.._n-1
      x = rand
      y = rand

      z1 = Math::sqrt(-2 * Math::log(x)) * Math::cos(2 * Math::PI * y)
      z2 = Math::sqrt(-2 * Math::log(x)) * Math::sin(2 * Math::PI * y)

      z1 *= Math::sqrt(@s)
      z2 *= Math::sqrt(@s)

      z1 += @u
      z2 += @u

      ss += [z1, z2]
    end

    ss.shuffle
  end

  #### likelihood ####
  def likelihood(_x)
    Math::exp(-(@u - _x)**2 / (2*@s) ) / Math::sqrt(2 * Math::PI * @s)
  end

  def log_likelihood(_x)
    -(@u - _x)**2 / (2*@s)  -  Math::log(2 * Math::PI * @s) / 2
  end

  def type
    "Gaussian: ave = #{@u}, var = #{s}"
  end
end

################################################################################
# Naive Bayes Model
################################################################################
class MyNaiveBayesModel
  #### new ####
  def initialize(_k, _n)
    @k = _k # # class
    @n = _n # # atribute
    @c = Array.new(@k){|k| Array.new(@n){|n| nil}}

    # mixing rate
    @p = Array.new(@k){|i| rand}
    sum = 0
    for i in 0..@k-1
      sum += @p[i]
    end
    @p.map!{|i| i /= sum}
  end

  #### def data type ####
  def set_component(_k, _n, _c)
    @c[_k][_n] = _c
  end

  #### init ####
  def init
    for n in 0..@n-1
      flag = true
      flag = false if rand < 0.5

      for k in 0..@k-1
        if flag
          @c[k][n] = MyCategorical.new(["a", "b", "c"])
        else
          @c[k][n] = MyGaussian.new(0, 1)
        end
      end
    end
  end

  #### load ####
  def MyNaiveBayesModel.new_from_data(_k, _d)
    _n = _d[0].size
    mm = MyNaiveBayesModel.new(_k, _n)

    # guess data type
    vals = Array.new(_n){|h| Hash.new(0)}
    _d.each do |x|
      for n in 0.._n-1
        vals[n][x[n]] += 1
      end
    end

    # set components
    for n in 0.._n-1
      # check type
      flag = numbers?(vals[n].keys)

      # convert string -> float
      if flag
        for i in 0.._d.size-1
          _d[i][n] = _d[i][n].to_f
        end
      end

      # define components
      for k in 0.._k-1
        if flag
          mm.set_component(k, n, MyGaussian.new(0, 1))
        else
          mm.set_component(k, n, MyCategorical.new(vals[n].keys))
        end
      end
    end

    mm
  end

  #### sample ####
  def sample(_n)
    ys = Array.new(_n){|i| sample_y}
    xs = Array.new(_n){|i| sample_x(ys[i])}

    [ys, xs]
  end
  def sample_y
    s = 0
    r = rand
    for i in 0..@k-1
      s += @p[i]
      return i if r < s
    end
  end
  def sample_x(_k)
    Array.new(@n){|i| @c[_k][i].sample(1)[0] }
  end

  #### learn ####
  def learn(_xs)
    step = 0
    n = _xs.size
    ys = Array.new(n){|i| rand(@k)}

    begin
      #### separate data ####
      d = Array.new(@k){|i| []}
      for i in 0..n-1
        y = ys[i]
        x = _xs[i]
        d[y].push(x)
      end

      #### learn mixing rate ####
      for k in 0..@k-1
        @p[k] = d[k].size / n.to_f
      end

      #### learn compornents ####
      for k in 0..@k-1
        learn_k(k, d[k]) if d[k].size > 0
      end

      #### viterbi ####
      flag = false
      for i in 0..n-1
        y = ys[i]       # old cluster
        x = _xs[i]      # data
        k = viterbi(x)  # new cluster
        flag = true if y != k
        ys[i] = k
      end
      step += 1 if flag
    end while flag || step > 1000

    step
  end

  #### learn k-th component ####
  def learn_k(_k, _d)
    for n in 0..@n-1
      @c[_k][n].learn( _d.map{|i| i[n]} )
    end
  end

  #### likelihood ####
  def log_likelihood(_xs)
    ld = 0
    _xs.each do |x|
      y = viterbi(x)
      ld += log_likelihood_k(x, y)
    end
    ld
  end
  def log_likelihood_k(_x, k)
    ld = 0
    for n in 0..@n-1
      ld += @c[k][n].log_likelihood(_x[n])
    end
    ld
  end

  #### viterbi ####
  def viterbi(_x)
    max_k = 0
    max_d = log_likelihood_k(_x, 0)

    for k in 1..@k-1
      if (d = log_likelihood_k(_x, k)) > max_d
        max_k = k
        max_d = d
      end
    end

    max_k
  end

  #### clustering ####
  def clustering(_xs)
    d = Array.new(@k){|i| [] }
    for i in 0.._xs.size-1
      x = _xs[i]
      d[ viterbi(x) ].push(i)
    end
    d
  end

  #### show ####
  def show
    for k in 0..@k-1
      puts "Class #{k+1} (#{@p[k]})"
      for n in 0..@n-1
        puts "#{n}-th attr: #{@c[k][n].type}"
      end
    end
  end
end

################################################################################
# fanctions
################################################################################
########################################
# scaling
########################################
def scale(_xs)
  for n in 0.._xs[0].size-1
    scale_n(_xs, n)
  end
end
def scale_n(_xs, _n)
  num = _xs.map{|x| x[_n]}
  return false if !numbers?(num)
  num.map!{|n| n.to_f}

  min = nil
  max = nil
  num.each do |n|
    min = n if min == nil || min > n
    max = n if max == nil || max < n
  end

  for i in 0.._xs.size-1
    _xs[i][_n] =  (_xs[i][_n].to_f - min) / (max - min).to_f
  end

  return true
end

########################################
# numbers or not
########################################
def numbers?(_ns)
  return false if _ns.size <= 10

  _ns.each do |n|
    return false if !(n.to_s =~ /\d+.\d+/ || n.to_s =~ /\d/)
  end

  return true
end

########################################
# generate random data
########################################
def rand_data(_m)
  sm = MyNaiveBayesModel.new(@k, @n)
  sm.init
  sm.sample(@m)
end

def load_data(_file, _y)
  id = 0
  ys = []
  xs = []

  open(_file).read.split("\n").each do |line|
    x = line.split(/[,;:]/)

    # target
    if _y != nil
      y = x[_y]
      x.delete_at(_y)
    else
      y = (id += 1)
    end

    # data
    ys.push(y)
    xs.push(x.clone)
  end

  [ys, xs]
end

################################################################################
# main
################################################################################

#### load data ####
if @file == nil
  db = rand_data(@m)
else
  db = load_data(@file, @y)
end

# ignored features
ys = db[0]
xs = db[1]
ds = []

xs.each do |x|
  ds.push(x.clone)
  @I.sort.reverse.each do |i|
    x.delete_at(i)
  end
end

puts "#### setting ####"
puts "# data     = #{xs.size}"
puts "# Features = #{@n}"
puts "# restart  = #{@i}"
puts "Target     = #{@y}" if @y != nil
puts "Ignored    = #{@I}" if @I.size > 0

#### learn ####
puts "#### learning ####"
scale(xs) if @scale

best_lm = nil
best_ld = nil
for i in 1..@i
  lm = MyNaiveBayesModel.new_from_data(@k, xs)
  s  = lm.learn(xs)
  ld = lm.log_likelihood(xs)

  printf("%3d : %5d steps  (%5.3e)\n", i, s, ld)

  if best_lm == nil || best_ld < ld
    best_lm = lm
    best_ld = ld
  end
end

#### result ####
puts "#### learned distribution ####"
best_lm.show

puts "#### clustering result ####"
h = Hash.new(0)
c = best_lm.clustering(xs)
for k in 0..@k-1
  h.clear

  c[k].each do |i|
    h[ ys[i] ] += 1
  end

  puts "cluster #{k+1} : #{c[k].size} #{h}"

  next if !@show
  c[k].each do |i|
    y = ys[i]
    x = xs[i]
    d = ds[i]
    puts "#{y} #{d.map{|i| sprintf("%s", i)}.join(", ")}"
  end
end
