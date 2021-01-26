# Mobile-Price-Prediction
Mobile Price Prediction Using Random Forest


{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 348,
   "metadata": {
    "id": "6S0eGWcHFtCa"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 349,
   "metadata": {
    "id": "zBYgk3d5FtCh"
   },
   "outputs": [],
   "source": [
    "df = pd.read_csv('train.csv')\n",
    "tdf = pd.read_csv('test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 350,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 224
    },
    "id": "NvbXYTvhFtCh",
    "outputId": "d41554de-60d7-4ce6-9985-369440db7120"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>battery_power</th>\n",
       "      <th>blue</th>\n",
       "      <th>clock_speed</th>\n",
       "      <th>dual_sim</th>\n",
       "      <th>fc</th>\n",
       "      <th>four_g</th>\n",
       "      <th>int_memory</th>\n",
       "      <th>m_dep</th>\n",
       "      <th>mobile_wt</th>\n",
       "      <th>n_cores</th>\n",
       "      <th>pc</th>\n",
       "      <th>px_height</th>\n",
       "      <th>px_width</th>\n",
       "      <th>ram</th>\n",
       "      <th>sc_h</th>\n",
       "      <th>sc_w</th>\n",
       "      <th>talk_time</th>\n",
       "      <th>three_g</th>\n",
       "      <th>touch_screen</th>\n",
       "      <th>wifi</th>\n",
       "      <th>price_range</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>842</td>\n",
       "      <td>0</td>\n",
       "      <td>2.2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>0.6</td>\n",
       "      <td>188</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>20</td>\n",
       "      <td>756</td>\n",
       "      <td>2549</td>\n",
       "      <td>9</td>\n",
       "      <td>7</td>\n",
       "      <td>19</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1021</td>\n",
       "      <td>1</td>\n",
       "      <td>0.5</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>53</td>\n",
       "      <td>0.7</td>\n",
       "      <td>136</td>\n",
       "      <td>3</td>\n",
       "      <td>6</td>\n",
       "      <td>905</td>\n",
       "      <td>1988</td>\n",
       "      <td>2631</td>\n",
       "      <td>17</td>\n",
       "      <td>3</td>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>563</td>\n",
       "      <td>1</td>\n",
       "      <td>0.5</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>41</td>\n",
       "      <td>0.9</td>\n",
       "      <td>145</td>\n",
       "      <td>5</td>\n",
       "      <td>6</td>\n",
       "      <td>1263</td>\n",
       "      <td>1716</td>\n",
       "      <td>2603</td>\n",
       "      <td>11</td>\n",
       "      <td>2</td>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>615</td>\n",
       "      <td>1</td>\n",
       "      <td>2.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>10</td>\n",
       "      <td>0.8</td>\n",
       "      <td>131</td>\n",
       "      <td>6</td>\n",
       "      <td>9</td>\n",
       "      <td>1216</td>\n",
       "      <td>1786</td>\n",
       "      <td>2769</td>\n",
       "      <td>16</td>\n",
       "      <td>8</td>\n",
       "      <td>11</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1821</td>\n",
       "      <td>1</td>\n",
       "      <td>1.2</td>\n",
       "      <td>0</td>\n",
       "      <td>13</td>\n",
       "      <td>1</td>\n",
       "      <td>44</td>\n",
       "      <td>0.6</td>\n",
       "      <td>141</td>\n",
       "      <td>2</td>\n",
       "      <td>14</td>\n",
       "      <td>1208</td>\n",
       "      <td>1212</td>\n",
       "      <td>1411</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>15</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   battery_power  blue  clock_speed  ...  touch_screen  wifi  price_range\n",
       "0            842     0          2.2  ...             0     1            1\n",
       "1           1021     1          0.5  ...             1     0            2\n",
       "2            563     1          0.5  ...             1     0            2\n",
       "3            615     1          2.5  ...             0     0            2\n",
       "4           1821     1          1.2  ...             1     0            1\n",
       "\n",
       "[5 rows x 21 columns]"
      ]
     },
     "execution_count": 350,
     "metadata": {
      "tags": []
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 351,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "1gCDtFlRFtCj",
    "outputId": "65285cfa-c1df-44a2-b582-0a40c1ad8b0e"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 2000 entries, 0 to 1999\n",
      "Data columns (total 21 columns):\n",
      " #   Column         Non-Null Count  Dtype  \n",
      "---  ------         --------------  -----  \n",
      " 0   battery_power  2000 non-null   int64  \n",
      " 1   blue           2000 non-null   int64  \n",
      " 2   clock_speed    2000 non-null   float64\n",
      " 3   dual_sim       2000 non-null   int64  \n",
      " 4   fc             2000 non-null   int64  \n",
      " 5   four_g         2000 non-null   int64  \n",
      " 6   int_memory     2000 non-null   int64  \n",
      " 7   m_dep          2000 non-null   float64\n",
      " 8   mobile_wt      2000 non-null   int64  \n",
      " 9   n_cores        2000 non-null   int64  \n",
      " 10  pc             2000 non-null   int64  \n",
      " 11  px_height      2000 non-null   int64  \n",
      " 12  px_width       2000 non-null   int64  \n",
      " 13  ram            2000 non-null   int64  \n",
      " 14  sc_h           2000 non-null   int64  \n",
      " 15  sc_w           2000 non-null   int64  \n",
      " 16  talk_time      2000 non-null   int64  \n",
      " 17  three_g        2000 non-null   int64  \n",
      " 18  touch_screen   2000 non-null   int64  \n",
      " 19  wifi           2000 non-null   int64  \n",
      " 20  price_range    2000 non-null   int64  \n",
      "dtypes: float64(2), int64(19)\n",
      "memory usage: 328.2 KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 352,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 317
    },
    "id": "2rWFNPaoFtCj",
    "outputId": "d796bd82-f275-46d0-b79d-25e429b87a47"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>battery_power</th>\n",
       "      <th>blue</th>\n",
       "      <th>clock_speed</th>\n",
       "      <th>dual_sim</th>\n",
       "      <th>fc</th>\n",
       "      <th>four_g</th>\n",
       "      <th>int_memory</th>\n",
       "      <th>m_dep</th>\n",
       "      <th>mobile_wt</th>\n",
       "      <th>n_cores</th>\n",
       "      <th>pc</th>\n",
       "      <th>px_height</th>\n",
       "      <th>px_width</th>\n",
       "      <th>ram</th>\n",
       "      <th>sc_h</th>\n",
       "      <th>sc_w</th>\n",
       "      <th>talk_time</th>\n",
       "      <th>three_g</th>\n",
       "      <th>touch_screen</th>\n",
       "      <th>wifi</th>\n",
       "      <th>price_range</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.0000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "      <td>2000.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>1238.518500</td>\n",
       "      <td>0.4950</td>\n",
       "      <td>1.522250</td>\n",
       "      <td>0.509500</td>\n",
       "      <td>4.309500</td>\n",
       "      <td>0.521500</td>\n",
       "      <td>32.046500</td>\n",
       "      <td>0.501750</td>\n",
       "      <td>140.249000</td>\n",
       "      <td>4.520500</td>\n",
       "      <td>9.916500</td>\n",
       "      <td>645.108000</td>\n",
       "      <td>1251.515500</td>\n",
       "      <td>2124.213000</td>\n",
       "      <td>12.306500</td>\n",
       "      <td>5.767000</td>\n",
       "      <td>11.011000</td>\n",
       "      <td>0.761500</td>\n",
       "      <td>0.503000</td>\n",
       "      <td>0.507000</td>\n",
       "      <td>1.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>439.418206</td>\n",
       "      <td>0.5001</td>\n",
       "      <td>0.816004</td>\n",
       "      <td>0.500035</td>\n",
       "      <td>4.341444</td>\n",
       "      <td>0.499662</td>\n",
       "      <td>18.145715</td>\n",
       "      <td>0.288416</td>\n",
       "      <td>35.399655</td>\n",
       "      <td>2.287837</td>\n",
       "      <td>6.064315</td>\n",
       "      <td>443.780811</td>\n",
       "      <td>432.199447</td>\n",
       "      <td>1084.732044</td>\n",
       "      <td>4.213245</td>\n",
       "      <td>4.356398</td>\n",
       "      <td>5.463955</td>\n",
       "      <td>0.426273</td>\n",
       "      <td>0.500116</td>\n",
       "      <td>0.500076</td>\n",
       "      <td>1.118314</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>501.000000</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.100000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>500.000000</td>\n",
       "      <td>256.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>851.750000</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>0.700000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>16.000000</td>\n",
       "      <td>0.200000</td>\n",
       "      <td>109.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>282.750000</td>\n",
       "      <td>874.750000</td>\n",
       "      <td>1207.500000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.750000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>1226.000000</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>1.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>32.000000</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>141.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>564.000000</td>\n",
       "      <td>1247.000000</td>\n",
       "      <td>2146.500000</td>\n",
       "      <td>12.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>11.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1615.250000</td>\n",
       "      <td>1.0000</td>\n",
       "      <td>2.200000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>48.000000</td>\n",
       "      <td>0.800000</td>\n",
       "      <td>170.000000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>15.000000</td>\n",
       "      <td>947.250000</td>\n",
       "      <td>1633.000000</td>\n",
       "      <td>3064.500000</td>\n",
       "      <td>16.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>16.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.250000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1998.000000</td>\n",
       "      <td>1.0000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>19.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>64.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>200.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>20.000000</td>\n",
       "      <td>1960.000000</td>\n",
       "      <td>1998.000000</td>\n",
       "      <td>3998.000000</td>\n",
       "      <td>19.000000</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>20.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       battery_power       blue  ...         wifi  price_range\n",
       "count    2000.000000  2000.0000  ...  2000.000000  2000.000000\n",
       "mean     1238.518500     0.4950  ...     0.507000     1.500000\n",
       "std       439.418206     0.5001  ...     0.500076     1.118314\n",
       "min       501.000000     0.0000  ...     0.000000     0.000000\n",
       "25%       851.750000     0.0000  ...     0.000000     0.750000\n",
       "50%      1226.000000     0.0000  ...     1.000000     1.500000\n",
       "75%      1615.250000     1.0000  ...     1.000000     2.250000\n",
       "max      1998.000000     1.0000  ...     1.000000     3.000000\n",
       "\n",
       "[8 rows x 21 columns]"
      ]
     },
     "execution_count": 352,
     "metadata": {
      "tags": []
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 353,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "xCv_XhbrSlxe",
    "outputId": "c31a18ba-bad3-4316-9ce9-31981f6f8755"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',\n",
       "       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',\n",
       "       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',\n",
       "       'touch_screen', 'wifi', 'price_range'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 353,
     "metadata": {
      "tags": []
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 354,
   "metadata": {
    "id": "XEGlRJlOFtCk"
   },
   "outputs": [],
   "source": [
    "X = df.iloc[:, :-1]\n",
    "y = df.iloc[:, -1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 355,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 333
    },
    "id": "lnXdtp3FalwE",
    "outputId": "415cd873-9330-457a-fa49-03eb6bbce986"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.0605788  0.01980083 0.03297645 0.01993833 0.03264442 0.01772261\n",
      " 0.03501928 0.03301766 0.03562312 0.03234857 0.03381531 0.04713539\n",
      " 0.04771047 0.39608445 0.03337411 0.03369078 0.03534147 0.0149126\n",
      " 0.01818518 0.02008015]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light",
      "tags": []
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.ensemble import ExtraTreesClassifier\n",
    "import matplotlib.pyplot as plt\n",
    "model = ExtraTreesClassifier()\n",
    "model.fit(X,y)\n",
    "print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers\n",
    "#plot graph of feature importances for better visualization\n",
    "feat_importances = pd.Series(model.feature_importances_, index=X.columns)\n",
    "feat_importances.nlargest(len(X.columns)).plot(kind='barh')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 356,
   "metadata": {
    "id": "5H9lnGsmcII0"
   },
   "outputs": [],
   "source": [
    "df = df[['battery_power', 'fc', 'int_memory', 'n_cores', 'pc', 'px_height', 'px_width', \n",
    "         'ram', 'sc_h', 'sc_w', 'touch_screen', 'price_range']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 357,
   "metadata": {
    "id": "tDNX9qnQFtCl"
   },
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "iwwrBUZChA7l"
   },
   "source": [
    "# Random Forest Classification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 358,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "lNwk3luUFtCl",
    "outputId": "e404742d-e923-4b2c-8d98-3e8507cbe803"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy for Random Forest  80  is:  86.25\n",
      "Accuracy for Random Forest  85  is:  86.25\n",
      "Accuracy for Random Forest  90  is:  86.25\n",
      "Accuracy for Random Forest  95  is:  87.0\n",
      "Accuracy for Random Forest  100  is:  86.75\n",
      "Accuracy for Random Forest  105  is:  87.25\n",
      "Accuracy for Random Forest  110  is:  87.25\n",
      "Accuracy for Random Forest  115  is:  87.5\n",
      "Accuracy for Random Forest  120  is:  87.0\n",
      "Accuracy for Random Forest  125  is:  87.5\n",
      "Accuracy for Random Forest  130  is:  87.5\n",
      "Accuracy for Random Forest  135  is:  87.0\n",
      "Accuracy for Random Forest  140  is:  86.75\n",
      "Accuracy for Random Forest  145  is:  87.25\n",
      "Accuracy for Random Forest  150  is:  87.0\n",
      "Accuracy for Random Forest  155  is:  86.5\n",
      "Accuracy for Random Forest  160  is:  86.75\n",
      "Accuracy for Random Forest  165  is:  86.5\n",
      "Accuracy for Random Forest  170  is:  87.0\n",
      "Accuracy for Random Forest  175  is:  86.5\n",
      "Accuracy for Random Forest  180  is:  86.75\n",
      "Accuracy for Random Forest  185  is:  86.5\n",
      "Accuracy for Random Forest  190  is:  86.5\n",
      "Accuracy for Random Forest  195  is:  86.5\n"
     ]
    }
   ],
   "source": [
    "accuracy_list = []\n",
    "esti_list = range(80, 200, 5)\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "for i in esti_list:\n",
    "  classifier = RandomForestClassifier(n_estimators = i, criterion = 'entropy', random_state = 0)\n",
    "  classifier.fit(X_train, y_train)\n",
    "  y_pred = classifier.predict(X_test)\n",
    "  accuracy = accuracy_score(y_test ,y_pred) * 100\n",
    "  accuracy_list.append(accuracy)\n",
    "  print (\"Accuracy for Random Forest \",i,\" is: \", accuracy) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 359,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 279
    },
    "id": "RC4kx8Vco4yY",
    "outputId": "565bf00d-ffe1-4581-b33e-213f7e677a4f"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light",
      "tags": []
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.subplots(figsize = (10, 4))\n",
    "plt.plot(esti_list, accuracy_list)\n",
    "plt.xlabel('Number of Estimators')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 360,
   "metadata": {
    "id": "fevbrMu_FtCm"
   },
   "outputs": [],
   "source": [
    "# from sklearn.metrics import confusion_matrix, accuracy_score\n",
    "# print(confusion_matrix(y_test, y_pred))\n",
    "# classifier.score(X_test, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "4DD_rzDshJSl"
   },
   "source": [
    "# Hyperparameter Tuning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 361,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "j9QhD5V0FtCo",
    "outputId": "02e8b753-9344-4a71-c61d-56b6d6b4f9ad"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [10, 120, 230, 340, 450, 560, 670, 780, 890, 1000], 'min_samples_split': [2, 5, 10, 14], 'min_samples_leaf': [1, 2, 4, 6, 8], 'criterion': ['entropy', 'gini']}\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import RandomizedSearchCV\n",
    "# Number of trees in random forest\n",
    "n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]\n",
    "# Number of features to consider at every split\n",
    "max_features = ['auto', 'sqrt','log2']\n",
    "# Maximum number of levels in tree\n",
    "max_depth = [int(x) for x in np.linspace(10, 1000,10)]\n",
    "# Minimum number of samples required to split a node\n",
    "min_samples_split = [2, 5, 10,14]\n",
    "# Minimum number of samples required at each leaf node\n",
    "min_samples_leaf = [1, 2, 4,6,8]\n",
    "# Create the random grid\n",
    "random_grid = {'n_estimators': n_estimators,\n",
    "               'max_features': max_features,\n",
    "               'max_depth': max_depth,\n",
    "               'min_samples_split': min_samples_split,\n",
    "               'min_samples_leaf': min_samples_leaf,\n",
    "              'criterion':['entropy','gini']}\n",
    "print(random_grid)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 362,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "A5BB9IDsMqfd",
    "outputId": "5e5468c1-04e1-43ec-d1e3-3df5eb67365f"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 3 folds for each of 20 candidates, totalling 60 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.\n",
      "[Parallel(n_jobs=-1)]: Done  37 tasks      | elapsed:  1.9min\n",
      "[Parallel(n_jobs=-1)]: Done  60 out of  60 | elapsed:  2.7min finished\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "RandomizedSearchCV(cv=3, error_score=nan,\n",
       "                   estimator=RandomForestClassifier(bootstrap=True,\n",
       "                                                    ccp_alpha=0.0,\n",
       "                                                    class_weight=None,\n",
       "                                                    criterion='gini',\n",
       "                                                    max_depth=None,\n",
       "                                                    max_features='auto',\n",
       "                                                    max_leaf_nodes=None,\n",
       "                                                    max_samples=None,\n",
       "                                                    min_impurity_decrease=0.0,\n",
       "                                                    min_impurity_split=None,\n",
       "                                                    min_samples_leaf=1,\n",
       "                                                    min_samples_split=2,\n",
       "                                                    min_weight_fraction_leaf=0.0,\n",
       "                                                    n_estimators=100,\n",
       "                                                    n_jobs...\n",
       "                   param_distributions={'criterion': ['entropy', 'gini'],\n",
       "                                        'max_depth': [10, 120, 230, 340, 450,\n",
       "                                                      560, 670, 780, 890,\n",
       "                                                      1000],\n",
       "                                        'max_features': ['auto', 'sqrt',\n",
       "                                                         'log2'],\n",
       "                                        'min_samples_leaf': [1, 2, 4, 6, 8],\n",
       "                                        'min_samples_split': [2, 5, 10, 14],\n",
       "                                        'n_estimators': [200, 400, 600, 800,\n",
       "                                                         1000, 1200, 1400, 1600,\n",
       "                                                         1800, 2000]},\n",
       "                   pre_dispatch='2*n_jobs', random_state=100, refit=True,\n",
       "                   return_train_score=False, scoring=None, verbose=2)"
      ]
     },
     "execution_count": 362,
     "metadata": {
      "tags": []
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rf=RandomForestClassifier()\n",
    "rf_randomcv=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=20,cv=3,verbose=2,\n",
    "                               random_state=100,n_jobs=-1)\n",
    "rf_randomcv.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 363,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "QT5NlJA4Mqoj",
    "outputId": "10c4c009-1aca-40fc-b6bb-f2c91ab79174"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'criterion': 'entropy',\n",
       " 'max_depth': 1000,\n",
       " 'max_features': 'sqrt',\n",
       " 'min_samples_leaf': 2,\n",
       " 'min_samples_split': 2,\n",
       " 'n_estimators': 1400}"
      ]
     },
     "execution_count": 363,
     "metadata": {
      "tags": []
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rf_randomcv.best_params_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 364,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "zouSU3vbMqwh",
    "outputId": "748612f6-aa3c-4899-ef20-a18311722bfd"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomizedSearchCV(cv=3, error_score=nan,\n",
       "                   estimator=RandomForestClassifier(bootstrap=True,\n",
       "                                                    ccp_alpha=0.0,\n",
       "                                                    class_weight=None,\n",
       "                                                    criterion='gini',\n",
       "                                                    max_depth=None,\n",
       "                                                    max_features='auto',\n",
       "                                                    max_leaf_nodes=None,\n",
       "                                                    max_samples=None,\n",
       "                                                    min_impurity_decrease=0.0,\n",
       "                                                    min_impurity_split=None,\n",
       "                                                    min_samples_leaf=1,\n",
       "                                                    min_samples_split=2,\n",
       "                                                    min_weight_fraction_leaf=0.0,\n",
       "                                                    n_estimators=100,\n",
       "                                                    n_jobs...\n",
       "                   param_distributions={'criterion': ['entropy', 'gini'],\n",
       "                                        'max_depth': [10, 120, 230, 340, 450,\n",
       "                                                      560, 670, 780, 890,\n",
       "                                                      1000],\n",
       "                                        'max_features': ['auto', 'sqrt',\n",
       "                                                         'log2'],\n",
       "                                        'min_samples_leaf': [1, 2, 4, 6, 8],\n",
       "                                        'min_samples_split': [2, 5, 10, 14],\n",
       "                                        'n_estimators': [200, 400, 600, 800,\n",
       "                                                         1000, 1200, 1400, 1600,\n",
       "                                                         1800, 2000]},\n",
       "                   pre_dispatch='2*n_jobs', random_state=100, refit=True,\n",
       "                   return_train_score=False, scoring=None, verbose=2)"
      ]
     },
     "execution_count": 364,
     "metadata": {
      "tags": []
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rf_randomcv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 365,
   "metadata": {
    "id": "DqPr-tSCMq3Y"
   },
   "outputs": [],
   "source": [
    "best_random_grid = rf_randomcv.best_estimator_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 366,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "xNvk2CgHMq8K",
    "outputId": "bc9ac864-a676-434d-fa3f-968426590850"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 94   1   0   0]\n",
      " [  5  74  13   0]\n",
      " [  0  18  73   8]\n",
      " [  0   0   9 105]]\n",
      "Accuracy Score 0.865\n",
      "Classification report:               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.95      0.99      0.97        95\n",
      "           1       0.80      0.80      0.80        92\n",
      "           2       0.77      0.74      0.75        99\n",
      "           3       0.93      0.92      0.93       114\n",
      "\n",
      "    accuracy                           0.86       400\n",
      "   macro avg       0.86      0.86      0.86       400\n",
      "weighted avg       0.86      0.86      0.86       400\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "y_pred=best_random_grid.predict(X_test)\n",
    "print(confusion_matrix(y_test,y_pred))\n",
    "print(\"Accuracy Score {}\".format(accuracy_score(y_test,y_pred)))\n",
    "print(\"Classification report: {}\".format(classification_report(y_test,y_pred)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7ssGYUjUFtCo"
   },
   "source": [
    "# Output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 367,
   "metadata": {
    "id": "Xpzde9SDFtCp"
   },
   "outputs": [],
   "source": [
    "tdf.drop('id', inplace=True, axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 368,
   "metadata": {
    "id": "OLYgzQ12FtCp"
   },
   "outputs": [],
   "source": [
    "y_pred = classifier.predict(tdf.to_numpy())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 369,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "9zdZ2a1CFtCp",
    "outputId": "7d509a52-8b2c-4eef-9cd6-244057fd1e08"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[3 3 2 3 1 3 3 1 3 0 3 3 0 0 2 0 1 1 3 2 1 3 1 1 3 0 2 0 2 0 2 0 3 0 0 1 3\n",
      " 1 2 1 1 2 0 0 0 1 0 3 1 2 2 0 2 0 3 1 3 1 1 3 3 2 0 1 1 1 1 3 1 1 1 2 2 3\n",
      " 3 0 2 0 2 3 0 3 3 0 3 0 3 1 3 0 1 1 2 0 2 1 0 2 1 2 1 0 0 3 1 2 0 1 2 3 3\n",
      " 2 1 3 3 3 3 1 3 0 0 3 2 1 1 0 3 2 3 1 0 1 1 1 3 1 1 0 3 2 1 3 1 3 2 3 3 2\n",
      " 2 3 2 3 0 0 2 2 3 3 3 3 2 2 3 3 3 3 1 0 3 0 0 0 1 1 0 1 0 0 1 2 1 0 0 0 2\n",
      " 2 2 1 0 0 0 1 0 3 1 0 2 2 2 3 1 2 3 3 3 1 2 1 0 0 1 3 0 2 3 3 0 2 0 3 2 2\n",
      " 3 0 0 1 0 3 0 1 0 2 2 1 3 0 2 0 3 1 2 0 0 2 1 3 3 3 1 1 3 0 0 2 3 3 1 3 1\n",
      " 1 3 2 1 2 3 3 3 1 0 1 2 3 1 1 3 2 0 3 0 1 2 0 0 3 2 3 2 2 1 3 3 2 3 2 2 1\n",
      " 1 0 2 3 1 0 0 3 0 3 0 1 2 0 2 3 1 3 2 2 1 2 0 0 0 1 3 2 0 0 0 3 1 0 3 3 1\n",
      " 2 3 2 3 1 3 3 2 2 2 3 3 0 3 0 3 1 3 1 3 3 0 1 1 3 1 3 1 3 0 0 0 0 2 0 0 2\n",
      " 1 1 2 3 2 0 1 0 0 3 2 0 3 1 2 2 1 2 3 1 1 2 2 1 2 0 1 1 0 3 2 0 0 1 0 0 1\n",
      " 0 0 0 0 2 2 3 2 3 0 3 0 3 0 1 1 1 2 0 3 2 3 3 1 3 1 3 1 3 2 1 2 2 1 1 0 0\n",
      " 0 1 2 1 0 3 2 0 2 3 0 0 3 1 1 0 2 3 3 0 3 0 2 3 2 3 0 2 0 2 3 0 1 1 0 0 1\n",
      " 1 2 3 3 3 2 3 1 1 2 3 3 3 1 0 2 2 2 2 1 0 2 2 0 0 0 3 1 1 2 2 2 0 3 0 2 2\n",
      " 0 3 0 2 3 0 1 1 3 3 1 1 2 3 2 0 2 1 2 0 3 3 1 2 2 2 3 0 1 2 3 1 3 2 3 1 0\n",
      " 1 0 3 1 0 3 2 3 2 0 3 3 3 2 3 3 1 1 0 2 3 3 1 0 1 1 2 2 1 0 0 2 2 3 2 0 2\n",
      " 1 3 3 0 1 3 1 2 1 0 0 0 2 1 0 1 1 2 2 0 2 2 1 0 3 0 0 3 2 0 0 0 0 0 3 0 3\n",
      " 1 3 2 1 3 2 0 1 1 3 2 2 2 0 3 0 2 0 2 0 0 1 1 1 2 1 3 1 3 2 1 1 3 2 0 1 2\n",
      " 0 3 3 0 2 1 1 2 0 3 2 0 3 2 3 0 0 3 0 1 2 3 2 1 2 2 1 2 3 0 1 1 2 2 1 0 0\n",
      " 1 0 0 3 0 1 1 0 1 1 1 3 0 3 2 3 0 0 1 2 2 1 0 1 2 0 1 1 0 0 3 3 0 3 1 2 3\n",
      " 0 1 0 2 2 0 3 1 0 3 0 1 0 3 3 3 2 3 0 3 2 0 1 0 3 3 2 0 2 1 3 1 0 3 2 0 3\n",
      " 1 1 1 1 1 3 1 1 1 2 0 0 1 2 0 2 0 0 0 0 3 3 3 3 0 1 2 1 1 0 0 2 1 0 2 0 3\n",
      " 2 2 1 2 0 1 1 3 0 0 3 1 3 0 0 2 3 2 1 2 2 1 0 0 3 3 0 3 0 0 0 2 2 1 2 0 3\n",
      " 2 1 2 3 3 0 1 1 2 1 2 2 0 1 3 1 1 3 1 2 3 1 1 1 2 3 3 0 2 3 0 2 3 2 2 2 3\n",
      " 2 0 1 2 0 2 1 1 2 2 2 1 2 0 0 1 3 1 0 1 1 3 1 0 0 3 2 2 3 0 3 3 2 1 3 0 1\n",
      " 3 1 2 1 2 3 2 0 3 0 2 3 0 3 1 3 3 1 0 2 3 1 0 1 1 2 1 3 0 2 2 0 2 3 2 3 0\n",
      " 2 1 1 2 2 3 3 0 2 1 2 1 3 0 1 3 0 1 0 0 3 2 2 0 0 0 0 3 2 3 3 0 0 1 1 0 2\n",
      " 2]\n"
     ]
    }
   ],
   "source": [
    "print(y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 369,
   "metadata": {
    "id": "LzwqBxSyFtCp"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 369,
   "metadata": {
    "id": "7z5Fmjh3FtCp"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 369,
   "metadata": {
    "id": "8OQFwon5FtCq"
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [],
   "name": "Untitled.ipynb",
   "provenance": [],
   "toc_visible": true
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
