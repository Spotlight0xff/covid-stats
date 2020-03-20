"""
MPL rendering of statistics for the COVID-19 outbreak.
"""
import datetime
import os
from typing import List, Dict
import csv

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd
from subprocess import check_output
import tqdm as tqdm

REPO_URL = "https://github.com/CSSEGISandData/COVID-19"
COVID_DATA = "covid-data"
CSV_DATA_FILE = "%s/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv" % COVID_DATA


class CovidTimeSeries:
  """
  Represents the COVID-19 dataset, from https://github.com/CSSEGISandData/COVID-19
  """
  def __init__(self, csv_file):
    self.csv_file = csv_file

  def all_countries(self) -> List[str]:
    """Returns a list of all lists in the dataset."""
    countries = set()
    with open(self.csv_file, newline='') as csv_fp:
      reader = csv.reader(csv_fp, delimiter=',', quotechar='"')
      next(reader)  # skip header
      for row in reader:
        country = row[1]
        countries.add(country)
    return list(countries)

  def all_confirmed_cases(self):
    """Returns a pd.DataFrame of all the timeseries data."""
    dates = self.all_sorted_dates()
    df = pd.read_csv(self.csv_file, infer_datetime_format=True, quotechar="\"")

    # sort columns
    def sort_cols(col):
      """Returns a datetime.date object for a given MM/DD/YY string."""
      if col.count("/") < 2:  # skip Country/Region etc.
        return datetime.date.fromtimestamp(0)
      col_split = [int(t) for t in col.split("/")]
      return datetime.date(year=2000+col_split[2], month=col_split[0], day=col_split[1])
    sorted_cols = sorted(df.columns, key=sort_cols)
    df = df.reindex(sorted_cols, axis=1)

    # rename columns so we can interpolate (date -> [0, 1, 2,...])
    columns = ["Province/State", "Country/Region", "Lat", "Long"] + list(range(len(dates)))
    df.rename(dict(zip(df.columns, columns)), axis="columns", inplace=True)
    return df

  def stats_for_country(self, country):
    """Outdated method for grabbing stats about each country."""
    country_dict = None
    with open(self.csv_file, newline='') as csv_fp:
      reader = csv.reader(csv_fp, dialect="excel")
      header = next(reader)
      data_headers = header[4:]  # List[str] of dates MM/DD/YY
      for row in reader:
        country_csv = row[1]
        if not country_csv.lower() == country.lower():
          continue
        data_row = [int(a) for a in row[4:]]
        if country_dict is None:
          country_dict = dict((zip(data_headers, data_row)))  # dict[str->str]
        else:
          for i, date in enumerate(country_dict.keys()):
            country_dict[date] += data_row[i]
      return country_dict

  def all_sorted_dates(self):
    """
    :return: list of DD/MM/YY dates
    :rtype list[str]
    """
    with open(self.csv_file, newline='') as csv_fp:
      reader = csv.reader(csv_fp, dialect="excel")
      header = next(reader)
      return header[4:]


def set_style(size, aspect=None):
  """
  Sets the matplotlib style according to the parameters.

  :param float aspect:
  :param size: width in inches
  :return:
  """
  import warnings
  # matplotlib warns about the font family,
  # but the rendering is done by latex.
  warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib",
                          message=r"findfont: Font family.*")
  dpi = 200
  # seaborn-paper: axes.titlesize: 9.6
  # seaborn-talk: axes.titlesize: 15.6
  assert isinstance(aspect, (int, float))
  figsize = (size, size * aspect)
  width = size  # [inches]
  titlesize = width / 5.5 * 11
  legendsize = labelsize = width / 5.5 * 10

  # noinspection PyTypeChecker
  # print(plt.style.available)
  # for font in matplotlib.font_manager.fontManager.ttflist:
  #   print(font)
  plt.style.use(['seaborn-whitegrid',
                 {
                   # 'text.usetex': True,
                   # 'text.latex.preamble': [r"\usepackage{times}"],
                   'font.family': 'Noto Sans',
                   'font.size': labelsize,
                   # 'legend.facecolor': 'white',
                   'figure.dpi': dpi,
                   'figure.figsize': figsize,
                   'figure.autolayout': True,
                   'figure.titlesize': titlesize,
                   'axes.titlesize': titlesize,
                   'axes.labelsize': titlesize,
                   'xtick.labelsize': labelsize,
                   'ytick.labelsize': labelsize,
                   'legend.fontsize': legendsize,
                 }])


def update_data(dir):
  """Either clones the repo or fetches latest version."""
  if not os.path.isdir(dir):
    r = check_output(["git", "clone",  REPO_URL, dir])
  else:
    r = check_output(["git", "pull"], cwd=dir)
  print(r.decode("utf8"))


def augment_dataframe(df, num_steps=3):
  """Interpolate between days. num_steps=10 -> 10x more points"""
  # df.insert()
  # 45 = num_dates
  old_cols = list(df.columns)
  num_dates = len(old_cols) - 1
  integer_vals = range(num_dates)
  np.set_printoptions(precision=3, suppress=True)
  print(num_dates)
  float_vals = np.linspace(0, num_dates, num=num_dates*num_steps+1, endpoint=True)
  interp_vals = list(set(float_vals) - set(integer_vals))
  for val in interp_vals:
    df.loc[:, val] = np.nan
  new_cols = [old_cols[0]] + list(float_vals)
  df = df[new_cols]

  # perform the actual interpolation only on the numeric values (no country column)
  counts_df = df.loc[:, list(float_vals)].astype('float64').transpose()
  df.iloc[:, 1:] = counts_df.interpolate(axis=0, method="pchip").transpose()
  return df


def plot_animated_line_counts(df: pd.DataFrame, num_interpolated_steps):
  """Plots an animated line for the confirmed COVID-19 cases."""
  top_k = 8
  num_countries = len(df)
  num_dates = len(df.columns)-1  # "country/region" col
  set_style(12, aspect=0.6)

  cmap = sns.color_palette("Set1", n_colors=num_countries//2)
  cmap += sns.color_palette("Set2", n_colors=num_countries//2+1+1)
  country_colors = {v: cmap[i] for i, v in enumerate(df["Country/Region"])}

  pbar = tqdm.tqdm(total=num_dates)
  text_labels = []

  def animate(i):
    """Animate one frame of the plot. `i` indexes into the columns of `df`."""
    # cols: ["Country", "1/22/20", "1/23/20", ...]
    pbar.update()
    cols = df.columns.tolist()
    now = cols[int(i+1)]
    dates_until_now = cols[1:int(i+2)]

    top_data = df.loc[:, ["Country/Region"] + dates_until_now].sort_values(by=now, ascending=False).head(top_k)
    top_data.set_index("Country/Region", inplace=True)
    top_data = top_data.transpose()

    to_remove = []
    lines = []
    countries = top_data.columns.tolist()
    for i, country in enumerate(countries):
      if top_data[country][now] < 10 or country == "China":
        to_remove.append(country)
        continue

      p,  = plt.plot(top_data[country], '-x', color=country_colors[country], label=country,
                     markevery=num_interpolated_steps, markersize=7)

      text_labels[i].set_x(now)
      text_labels[i].set_y(top_data[country][now])
      text_labels[i].set_text(country)
      text_labels[i].set_color(country_colors[country])

      lines.append(p)

    # maintain order
    for country in to_remove:
      countries.remove(country)
    assert len(countries) == len(lines)
    ax.legend(lines, countries, loc="upper left")

    plt.setp(lines, linewidth=1)
    ax.set_ylabel("Confirmed COVID-19 cases")
    ax.set_title("Corona Virus propagation history")
    ax.set_xlabel("Time since outbreak (days)")

  print("Animating over %d frames." % num_dates)
  fig, ax = plt.subplots()
  for i in range(top_k):
    text_labels.append(ax.text(0, 0, "", fontsize=16))

  ani = FuncAnimation(fig, animate, frames=num_dates, repeat=False, blit=False)

  ffmpeg_writer = matplotlib.animation.writers['ffmpeg']
  writer = ffmpeg_writer(fps=25, metadata=dict(artist='Me'), bitrate=5000)
  ani.save('covid19_stats_countries.mp4', writer=writer)
  pbar.close()


def transform_aggregate_countries(df):
  """Aggregates data over countries."""
  index_cols = df.columns.tolist()
  index_cols.remove("Province/State")
  index_cols.remove("Country/Region")
  index_cols.remove("Lat")
  index_cols.remove("Long")
  df = df.groupby(["Country/Region"])[index_cols].apply(sum)
  df = df.reset_index()
  return df


def main():
  """Shows an animated plot for confirmed COVID-19 cases."""
  update_data(COVID_DATA)
  covid_timeseries = CovidTimeSeries(CSV_DATA_FILE)
  num_interpolated_steps = 20

  counts_df = covid_timeseries.all_confirmed_cases()
  counts_df = transform_aggregate_countries(counts_df)
  counts_df = augment_dataframe(counts_df, num_steps=num_interpolated_steps)
  plot_animated_line_counts(counts_df, num_interpolated_steps=num_interpolated_steps)


if __name__ == "__main__":
  import better_exchook
  better_exchook.install()
  main()
