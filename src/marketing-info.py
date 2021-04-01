# (\d{1,2})    (.*?)\n
# ### $1. $2\n

# "([0-9]*) (.*?)"
# ($1, "$2")

column_names = {
  1: "ANNUAL INCOME OF HOUSEHOLD (PERSONAL INCOME IF SINGLE)",
  2: "SEX",
  3: "MARITAL STATUS",
  4: "AGE",
  5: "EDUCATION",
  6: "OCCUPATION",
  7: "HOW LONG HAVE YOU LIVED IN THE SAN FRAN./OAKLAND/SAN JOSE AREA?",
  8: "DUAL INCOMES (IF MARRIED)",
  9: "PERSONS IN YOUR HOUSEHOLD",
  10: "PERSONS IN HOUSEHOLD UNDER 18",
  11: "HOUSEHOLDER STATUS",
  12: "TYPE OF HOME",
  13: "ETHNIC CLASSIFICATION",
  14: "WHAT LANGUAGE IS SPOKEN MOST OFTEN IN YOUR HOME?"
}

column_ranges = {
  1: {
    1: 'Less than $10,000',
    2: '$10,000 to $14,999',
    3: '$15,000 to $19,999',
    4: '$20,000 to $24,999',
    5: '$25,000 to $29,999',
    6: '$30,000 to $39,999',
    7: '$40,000 to $49,999',
    8: '$50,000 to $74,999',
    9: '$75,000 or more'
  },
  
  2: {
    1: 'Male',
    2: 'Female'
  },
  
  3: {
    1: 'Married',
    2: 'Living together, not married',
    3: 'Divorced or separated',
    4: 'Widowed',
    5: 'Single, never married'
  },
  
  4: {
    1: '14 thru 17',
    2: '18 thru 24',
    3: '25 thru 34',
    4: '35 thru 44',
    5: '45 thru 54',
    6: '55 thru 64',
    7: '65 and Over'
  },
  
  5: {
    1: 'Grade 8 or less',
    2: 'Grades 9 to 11',
    3: 'Graduated high school',
    4: '1 to 3 years of college',
    5: 'College graduate',
    6: 'Grad Study'
  },
  
  6: {
    1: 'Professional/Managerial',
    2: 'Sales Worker',
    3: 'Factory Worker/Laborer/Driver',
    4: 'Clerical/Service Worker',
    5: 'Homemaker',
    6: 'Student, HS or College',
    7: 'Military',
    8: 'Retired',
    9: 'Unemployed'
  },
  
  7: {
    1: 'Less than one year',
    2: 'One to three years',
    3: 'Four to six years',
    4: 'Seven to ten years',
    5: 'More than ten years'
  },
  
  8: {
    1: 'Not Married',
    2: 'Yes',
    3: 'No'
  },
  
  9: {
    1: 'One',
    2: 'Two',
    3: 'Three',
    4: 'Four',
    5: 'Five',
    6: 'Six',
    7: 'Seven',
    8: 'Eight',
    9: 'Nine or more'
  },
  
  10: {
    0: 'None',
    1: 'One',
    2: 'Two',
    3: 'Three',
    4: 'Four',
    5: 'Five',
    6: 'Six',
    7: 'Seven',
    8: 'Eight',
    9: 'Nine or more'
  },
  
  11: {
    1: 'Own',
    2: 'Rent',
    3: 'Live with Parents/Family',
  },
  
  12: {
    1: 'House',
    2: 'Condominium',
    3: 'Apartment',
    4: 'Mobile Home',
    5: 'Other',
  },
  13: {
    1: 'American Indian',
    2: 'Asian',
    3: 'Black',
    4: 'East Indian',
    5: 'Hispanic',
    6: 'Pacific Islander',
    7: 'White',
    8: 'Other',
  },
  
  14: {
    1: 'English',
    2: 'Spanish',
    3: 'Other',
  },
}